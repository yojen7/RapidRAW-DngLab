use crate::{
  RawImage, RawImageData, RawlerError, Result,
  formats::tiff::{IFD, Rational, Value},
  tags::DngTag,
};

fn invalid(message: &str) -> RawlerError {
  RawlerError::DecoderFailed(format!("DNG OpcodeList2: {message}"))
}

#[derive(Debug)]
pub(super) struct MapPolynomial {
  top: usize,
  left: usize,
  bottom: usize,
  right: usize,
  plane: usize,
  planes: usize,
  row_pitch: usize,
  col_pitch: usize,
  coefficients: Vec<f64>,
}

pub(super) fn mappings(ifd: &IFD) -> Result<Vec<MapPolynomial>> {
  match ifd.get_entry(DngTag::OpcodeList2) {
    None => Ok(Vec::new()),
    Some(entry) => match &entry.value {
      Value::Undefined(bytes) | Value::Byte(bytes) => parse(bytes),
      _ => Err(invalid("opcode list must contain bytes")),
    },
  }
}

fn parse(bytes: &[u8]) -> Result<Vec<MapPolynomial>> {
  fn u32_at(bytes: &[u8], offset: usize) -> Result<u32> {
    let data = bytes.get(offset..offset + 4).ok_or_else(|| invalid("truncated opcode list"))?;
    Ok(u32::from_be_bytes(data.try_into().unwrap()))
  }
  let count = u32_at(bytes, 0)? as usize;
  if count > bytes.len().saturating_sub(4) / 16 {
    return Err(invalid("opcode count exceeds list length"));
  }
  let mut offset = 4;
  let mut maps = Vec::new();
  let mut unsupported_required = false;
  for _ in 0..count {
    let id = u32_at(bytes, offset)?;
    let version = u32_at(bytes, offset + 4)?;
    let flags = u32_at(bytes, offset + 8)?;
    let size = u32_at(bytes, offset + 12)? as usize;
    offset += 16;
    let end = offset.checked_add(size).ok_or_else(|| invalid("opcode size overflow"))?;
    let payload = bytes.get(offset..end).ok_or_else(|| invalid("truncated opcode payload"))?;
    offset = end;
    if id == 8 && version > 0x0107_0100 && flags & 1 == 0 {
      return Err(invalid("MapPolynomial requires a newer DNG version"));
    }
    if id != 8 || version > 0x0107_0100 {
      unsupported_required |= flags & 1 == 0;
      continue;
    }
    let params: Vec<_> = (0..9).map(|i| u32_at(payload, i * 4)).collect::<Result<_>>()?;
    let degree = params[8] as usize;
    if degree > 8 || size != 36 + (degree + 1) * 8 {
      return Err(invalid("invalid MapPolynomial degree or payload length"));
    }
    if params[0] > params[2] || params[1] > params[3] || params[5] == 0 || params[6] == 0 || params[7] == 0 {
      return Err(invalid("invalid MapPolynomial area, planes or pitch"));
    }
    let coefficients: Vec<_> = payload[36..].chunks_exact(8).map(|b| f64::from_be_bytes(b.try_into().unwrap())).collect();
    if coefficients.iter().any(|c| !c.is_finite()) {
      return Err(invalid("nonfinite MapPolynomial coefficient"));
    }
    maps.push(MapPolynomial {
      top: params[0] as usize,
      left: params[1] as usize,
      bottom: params[2] as usize,
      right: params[3] as usize,
      plane: params[4] as usize,
      planes: params[5] as usize,
      row_pitch: params[6] as usize,
      col_pitch: params[7] as usize,
      coefficients,
    });
  }
  if offset != bytes.len() {
    return Err(invalid("trailing bytes in opcode list"));
  }
  // Preserve existing behavior for lists we cannot process at all, but never apply
  // only part of a required sequence alongside a supported nonlinear mapping.
  if !maps.is_empty() && unsupported_required {
    return Err(invalid("required opcode alongside MapPolynomial is unsupported"));
  }
  Ok(maps)
}

pub(super) fn apply_stage2(ifd: &IFD, image: &mut RawImage) -> Result<()> {
  let maps = mappings(ifd)?;
  if maps.is_empty() {
    return Ok(());
  }
  // DNG stage 2 is after linearization/black subtraction/scaling, before demosaic.
  // Return normalized Float32 pixels with matching black=0 / white=1 metadata.
  normalize(image)?;
  let RawImageData::Float(data) = &mut image.data else {
    return Err(invalid("stage 2 needs float pixels"));
  };
  apply(&maps, data, image.width, image.height, image.cpp)
}

fn normalize(image: &mut RawImage) -> Result<()> {
  let black = &image.blacklevel;
  let length = image
    .width
    .checked_mul(image.height)
    .and_then(|v| v.checked_mul(image.cpp))
    .ok_or_else(|| invalid("image dimensions overflow"))?;
  if image.cpp == 0
    || black.width == 0
    || black.height == 0
    || black.cpp != image.cpp
    || black.width.checked_mul(black.height).and_then(|v| v.checked_mul(black.cpp)) != Some(black.levels.len())
    || !(image.whitelevel.0.len() == 1 || image.whitelevel.0.len() == image.cpp)
  {
    return Err(invalid("invalid black/white level dimensions"));
  }
  let levels = black.as_vec();
  if levels
    .iter()
    .enumerate()
    .any(|(i, b)| !b.is_finite() || image.whitelevel.0[(i % image.cpp).min(image.whitelevel.0.len() - 1)] as f32 <= *b)
  {
    return Err(invalid("invalid black/white level range"));
  }
  let mut data = image.data.as_f32().into_owned();
  if data.len() != length {
    return Err(invalid("invalid image buffer"));
  }
  let (left, top) = image.active_area.map_or((0, 0), |area| (area.p.x, area.p.y));
  for (i, value) in data.iter_mut().enumerate() {
    if !value.is_finite() {
      return Err(invalid("nonfinite input pixel"));
    }
    let plane = i % image.cpp;
    let x = (i / image.cpp) % image.width;
    let y = (i / image.cpp) / image.width;
    let bx = (x + black.width - left % black.width) % black.width;
    let by = (y + black.height - top % black.height) % black.height;
    let b = levels[(by * black.width + bx) * image.cpp + plane];
    let w = image.whitelevel.0[plane.min(image.whitelevel.0.len() - 1)] as f32;
    *value = ((*value - b) / (w - b)).max(0.0);
  }
  image.data = RawImageData::Float(data);
  image.blacklevel.levels.iter_mut().for_each(|level| *level = Rational::new(0, 1));
  image.whitelevel.0.fill(1);
  Ok(())
}

fn apply(maps: &[MapPolynomial], data: &mut [f32], width: usize, height: usize, cpp: usize) -> Result<()> {
  let length = width
    .checked_mul(height)
    .and_then(|v| v.checked_mul(cpp))
    .ok_or_else(|| invalid("image dimensions overflow"))?;
  if cpp == 0 || data.len() != length {
    return Err(invalid("invalid image buffer"));
  }
  for map in maps {
    let last_plane = map.plane.saturating_add(map.planes).min(cpp);
    for row in (map.top..map.bottom.min(height)).step_by(map.row_pitch) {
      for col in (map.left..map.right.min(width)).step_by(map.col_pitch) {
        for plane in map.plane..last_plane {
          let pixel = &mut data[(row * width + col) * cpp + plane];
          if !pixel.is_finite() {
            return Err(invalid("nonfinite input pixel"));
          }
          let mapped = map.coefficients.iter().rev().fold(0.0, |value, c| value * f64::from(*pixel) + c);
          if mapped.is_nan() {
            return Err(invalid("invalid polynomial result"));
          }
          *pixel = mapped.clamp(0.0, 1.0) as f32;
        }
      }
    }
  }
  Ok(())
}

#[cfg(test)]
mod tests {
  use super::*;

  fn opcode(id: u32, flags: u32, area: [u32; 8], coefficients: &[f64]) -> Vec<u8> {
    let mut payload = Vec::new();
    for value in area.into_iter().chain([coefficients.len() as u32 - 1]) {
      payload.extend_from_slice(&value.to_be_bytes());
    }
    for c in coefficients {
      payload.extend_from_slice(&c.to_be_bytes());
    }
    let mut bytes = Vec::new();
    for value in [id, 0x0103_0000, flags, payload.len() as u32] {
      bytes.extend_from_slice(&value.to_be_bytes());
    }
    bytes.extend(payload);
    bytes
  }
  fn list(opcodes: &[Vec<u8>]) -> Vec<u8> {
    let mut bytes = (opcodes.len() as u32).to_be_bytes().to_vec();
    bytes.extend(opcodes.iter().flatten());
    bytes
  }
  #[test]
  fn sequential_mappings_respect_plane_and_clip_after_each_opcode() {
    let area = [0, 0, 1, 2, 1, 1, 1, 1];
    let maps = parse(&list(&[opcode(8, 0, area, &[0., 0., 4.]), opcode(8, 0, area, &[0., 0.5])])).unwrap();
    let mut data = [0.25, 0.5, 0.75, 0.25, 1., 0.75];
    apply(&maps, &mut data, 2, 1, 3).unwrap();
    assert_eq!(data, [0.25, 0.5, 0.75, 0.25, 0.5, 0.75]);
  }
  #[test]
  fn bayer_area_pitch_and_outside_image_do_not_touch_other_pixels() {
    let maps = parse(&list(&[opcode(8, 0, [1, 0, 9, 9, 0, 1, 2, 2], &[0.])])).unwrap();
    let mut data = [0.5; 16];
    apply(&maps, &mut data, 4, 4, 1).unwrap();
    let changed: Vec<_> = data.iter().enumerate().filter_map(|(i, v)| (*v == 0.).then_some(i)).collect();
    assert_eq!(changed, [4, 6, 12, 14]);
  }
  #[test]
  fn stage2_uses_black_white_levels_not_encoded_bit_depth() {
    use crate::{
      decoders::Camera,
      rawimage::{BlackLevel, RawPhotometricInterpretation, WhiteLevel},
    };
    let mut image = RawImage::new_with_data(
      Camera::new(),
      RawImageData::Integer(vec![600, 1200, 1800, 1100, 2200, 3300]),
      6,
      1,
      3,
      [1.; 4],
      RawPhotometricInterpretation::LinearRaw,
      Some(BlackLevel::new(&[100u16, 200, 300], 1, 1, 3)),
      Some(WhiteLevel::new(vec![1100, 2200, 3300])),
      false,
    );
    normalize(&mut image).unwrap();
    assert_eq!(image.data.as_f32().as_ref(), &[0.5, 0.5, 0.5, 1., 1., 1.]);
    assert_eq!(image.blacklevel.as_vec(), [0.; 3]);
    assert_eq!(image.whitelevel.0, [1; 3]);
    let maps = parse(&list(&[opcode(8, 0, [0, 0, 1, 2, 0, 3, 1, 1], &[0., 0., 1.])])).unwrap();
    let RawImageData::Float(ref mut pixels) = image.data else {
      panic!("normalized float data required");
    };
    apply(&maps, pixels, 2, 1, 3).unwrap();
    assert_eq!(pixels, &[0.25, 0.25, 0.25, 1., 1., 1.]);
  }
  #[test]
  fn unsupported_required_sequence_cannot_be_partially_applied() {
    let area = [0, 0, 1, 1, 0, 1, 1, 1];
    assert!(parse(&list(&[opcode(9, 0, area, &[1.]), opcode(8, 0, area, &[0.])])).is_err());
    assert!(parse(&list(&[opcode(9, 1, area, &[1.]), opcode(8, 0, area, &[0.])])).is_ok());
    assert!(parse(&list(&[opcode(9, 0, area, &[1.])])).unwrap().is_empty());
  }
  #[test]
  fn malformed_payloads_reject_without_panics_or_crossing_opcode_boundaries() {
    let good = list(&[opcode(8, 0, [0, 0, 1, 1, 0, 1, 1, 1], &[0., 1.])]);
    for n in 0..good.len() {
      assert!(parse(&good[..n]).is_err());
    }
    let mut bad = good.clone();
    bad[16..20].copy_from_slice(&u32::MAX.to_be_bytes());
    assert!(parse(&bad).is_err());
    let mut bad = good.clone();
    bad[16..20].copy_from_slice(&36u32.to_be_bytes());
    assert!(parse(&bad).is_err());
    assert!(parse(&list(&[opcode(8, 0, [0, 0, 1, 1, 0, 1, 0, 1], &[0.])])).is_err());
    assert!(parse(&list(&[opcode(8, 0, [0, 0, 1, 1, 0, 1, 1, 1], &[f64::NAN])])).is_err());
    assert!(parse(&list(&[opcode(8, 0, [0, 0, 1, 1, 0, 1, 1, 1], &[0.; 10])])).is_err());
  }
}
