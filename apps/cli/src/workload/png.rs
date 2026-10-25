//! Tiny PNG generator for the vision workload (256×256 RGB, stored deflate).
//!
//! No image codec dependency: the workload only needs a few distinct solid
//! images, and a stored-deflate PNG is deterministic and cheap to build.

const SIG: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

fn crc32(data: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for &b in data {
        crc ^= b as u32;
        for _ in 0..8 {
            let mask = (crc & 1).wrapping_neg();
            crc = (crc >> 1) ^ (0xedb8_8320 & mask);
        }
    }
    !crc
}

fn adler32(data: &[u8]) -> u32 {
    let mut a = 1u32;
    let mut b = 0u32;
    for &x in data {
        a = (a + x as u32) % 65_521;
        b = (b + a) % 65_521;
    }
    (b << 16) | a
}

fn chunk(kind: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(12 + data.len());
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(kind);
    out.extend_from_slice(data);
    let mut crc_in = Vec::with_capacity(4 + data.len());
    crc_in.extend_from_slice(kind);
    crc_in.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_in).to_be_bytes());
    out
}

fn stored_zlib(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(raw.len() + raw.len() / 65_535 * 5 + 8);
    out.extend_from_slice(&[0x78, 0x01]); // zlib header: deflate, 32K window
    let mut rest = raw;
    loop {
        let take = rest.len().min(65_535);
        let last = take == rest.len();
        out.push(if last { 0x01 } else { 0x00 }); // BFINAL | BTYPE=00 stored
        let len = take as u16;
        out.extend_from_slice(&len.to_le_bytes());
        out.extend_from_slice(&(!len).to_le_bytes());
        out.extend_from_slice(&rest[..take]);
        rest = &rest[take..];
        if last {
            break;
        }
    }
    out.extend_from_slice(&adler32(raw).to_be_bytes());
    out
}

/// 256×256 RGB PNG with a subtle vertical stripe pattern (vision-friendly).
pub fn png_rgb(w: u32, h: u32, rgb: (u8, u8, u8)) -> Vec<u8> {
    let mut raw = Vec::with_capacity((w as usize * 3 + 1) * h as usize);
    for y in 0..h {
        raw.push(0); // filter: None
        let stripe = 1.0 + 0.16 * (((y / 32) % 2) as f32);
        for x in 0..w {
            let shade = if x < w / 2 { 1.0 } else { 0.92 };
            let f = stripe * shade;
            raw.push((rgb.0 as f32 * f).min(255.0) as u8);
            raw.push((rgb.1 as f32 * f).min(255.0) as u8);
            raw.push((rgb.2 as f32 * f).min(255.0) as u8);
        }
    }
    let mut ihdr = Vec::with_capacity(13);
    ihdr.extend_from_slice(&w.to_be_bytes());
    ihdr.extend_from_slice(&h.to_be_bytes());
    ihdr.extend_from_slice(&[8, 2, 0, 0, 0]); // 8-bit RGB
    let idat = stored_zlib(&raw);
    let mut out = Vec::with_capacity(SIG.len() + 25 + idat.len() + 12 + 12);
    out.extend_from_slice(&SIG);
    out.extend_from_slice(&chunk(b"IHDR", &ihdr));
    out.extend_from_slice(&chunk(b"IDAT", &idat));
    out.extend_from_slice(&chunk(b"IEND", &[]));
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity((data.len() + 2) / 3 * 4);
    for c in data.chunks(3) {
        let b0 = c[0] as u32;
        let b1 = c.get(1).copied().unwrap_or(0) as u32;
        let b2 = c.get(2).copied().unwrap_or(0) as u32;
        let n = (b0 << 16) | (b1 << 8) | b2;
        out.push(T[((n >> 18) & 63) as usize] as char);
        out.push(T[((n >> 12) & 63) as usize] as char);
        out.push(if c.len() > 1 { T[((n >> 6) & 63) as usize] as char } else { '=' });
        out.push(if c.len() > 2 { T[(n & 63) as usize] as char } else { '=' });
    }
    out
}

pub fn data_url_png(w: u32, h: u32, rgb: (u8, u8, u8)) -> String {
    format!("data:image/png;base64,{}", base64(&png_rgb(w, h, rgb)))
}

pub const RED: (u8, u8, u8) = (210, 40, 40);
pub const BLUE: (u8, u8, u8) = (40, 80, 210);
pub const GREEN: (u8, u8, u8) = (40, 170, 80);
pub const YELLOW: (u8, u8, u8) = (220, 180, 40);
pub const PURPLE: (u8, u8, u8) = (160, 60, 200);
pub const CYAN: (u8, u8, u8) = (40, 190, 190);
