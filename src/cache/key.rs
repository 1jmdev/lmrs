pub fn hash_text(text: &str) -> u64 {
    let digest = blake3::hash(text.as_bytes());
    let bytes = digest.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}
