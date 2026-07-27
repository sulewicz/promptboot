use crate::gguf::{self, Value};
use crate::inspect::{self, InspectionReport};
use promptboot_core::sha256::Sha256;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

const HEADER_BYTES: u64 = 256;
const DIRECTORY_BYTES: u64 = 7 * 64;
const CEILING: u64 = 448 * 1024 * 1024;

pub enum PackFailure {
    Identity(String),
    Schema(String),
    Pack(String),
}

fn align(value: u64) -> Result<u64, String> {
    value
        .checked_add(63)
        .map(|result| result & !63)
        .ok_or_else(|| "container alignment overflow".to_owned())
}

fn hash_region(file: &mut File, offset: u64, length: u64) -> Result<[u8; 32], String> {
    file.seek(SeekFrom::Start(offset))
        .map_err(|error| format!("cannot seek while hashing: {error}"))?;
    let mut remaining = length;
    let mut buffer = vec![0; 1024 * 1024];
    let mut digest = Sha256::new();
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        let count = file
            .read(&mut buffer[..wanted])
            .map_err(|error| format!("cannot read while hashing: {error}"))?;
        if count == 0 {
            return Err(format!(
                "short read hashing region offset={offset} length={length}"
            ));
        }
        digest.update(&buffer[..count]);
        remaining -= count as u64;
    }
    Ok(digest.finish())
}

fn copy_region(
    source: &mut File,
    output: &mut File,
    offset: u64,
    length: u64,
    digest: &mut Sha256,
) -> Result<(), String> {
    source
        .seek(SeekFrom::Start(offset))
        .map_err(|error| format!("cannot seek source tensor: {error}"))?;
    let mut remaining = length;
    let mut buffer = vec![0; 1024 * 1024];
    while remaining != 0 {
        let wanted = usize::try_from(remaining.min(buffer.len() as u64)).unwrap();
        source
            .read_exact(&mut buffer[..wanted])
            .map_err(|error| format!("source ended while copying tensor: {error}"))?;
        output
            .write_all(&buffer[..wanted])
            .map_err(|error| format!("cannot write tensor: {error}"))?;
        digest.update(&buffer[..wanted]);
        remaining -= wanted as u64;
    }
    Ok(())
}

fn tokenizer_sections(metadata: &BTreeMap<String, Value>) -> Result<[Vec<u8>; 5], String> {
    let Value::ArrayString(tokens) = metadata
        .get("tokenizer.ggml.tokens")
        .ok_or_else(|| "missing token strings".to_owned())?
    else {
        return Err("token strings have wrong type".to_owned());
    };
    let Value::ArrayI32(token_types) = metadata
        .get("tokenizer.ggml.token_type")
        .ok_or_else(|| "missing token types".to_owned())?
    else {
        return Err("token types have wrong type".to_owned());
    };
    let Value::ArrayString(merges) = metadata
        .get("tokenizer.ggml.merges")
        .ok_or_else(|| "missing merges".to_owned())?
    else {
        return Err("merges have wrong type".to_owned());
    };
    let Value::ScalarString(template) = metadata
        .get("tokenizer.chat_template")
        .ok_or_else(|| "missing chat template".to_owned())?
    else {
        return Err("chat template has wrong type".to_owned());
    };

    let mut ids = BTreeMap::new();
    for (index, token) in tokens.iter().enumerate() {
        if token.is_empty() || ids.insert(token.as_str(), index as u32).is_some() {
            return Err("token strings must be unique and nonempty".to_owned());
        }
    }
    if ids.len() != 151936 {
        return Err("token count mismatch".to_owned());
    }
    let mut offsets = Vec::with_capacity((tokens.len() + 1) * 4);
    offsets.extend(0u32.to_le_bytes());
    let mut token_bytes = Vec::with_capacity(1_372_758);
    let mut longest = 0;
    for token in tokens {
        let encoded = token.as_bytes();
        longest = longest.max(encoded.len());
        token_bytes.extend(encoded);
        let end = u32::try_from(token_bytes.len())
            .map_err(|_| "token byte offsets exceed u32".to_owned())?;
        offsets.extend(end.to_le_bytes());
    }
    if token_bytes.len() != 1_372_758 || longest != 256 {
        return Err("token byte accounting mismatch".to_owned());
    }
    let mut type_bytes = Vec::with_capacity(token_types.len());
    for value in token_types {
        type_bytes.push(
            u8::try_from(*value).map_err(|_| format!("token type does not fit u8: {value}"))?,
        );
    }
    let mut merge_bytes = Vec::with_capacity(merges.len() * 12);
    let mut seen = BTreeSet::new();
    for (rank, merge) in merges.iter().enumerate() {
        let Some((left, right)) = merge.split_once(' ') else {
            return Err(format!("invalid source merge at rank {rank}"));
        };
        if right.contains(' ') {
            return Err(format!("invalid source merge at rank {rank}"));
        }
        let Some(&left_id) = ids.get(left) else {
            return Err(format!("unknown left merge token at rank {rank}"));
        };
        let Some(&right_id) = ids.get(right) else {
            return Err(format!("unknown right merge token at rank {rank}"));
        };
        if !seen.insert((left_id, right_id)) {
            return Err(format!("duplicate merge pair at rank {rank}"));
        }
        let mut combined = String::with_capacity(left.len() + right.len());
        combined.push_str(left);
        combined.push_str(right);
        let Some(&result_id) = ids.get(combined.as_str()) else {
            return Err(format!("merge result has no token at rank {rank}"));
        };
        merge_bytes.extend(left_id.to_le_bytes());
        merge_bytes.extend(right_id.to_le_bytes());
        merge_bytes.extend(result_id.to_le_bytes());
    }
    Ok([
        offsets,
        token_bytes,
        type_bytes,
        merge_bytes,
        template.as_bytes().to_vec(),
    ])
}

fn put_u32(buffer: &mut [u8], offset: usize, value: u32) {
    buffer[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

fn put_u64(buffer: &mut [u8], offset: usize, value: u64) {
    buffer[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
}

fn header(file_size: u64, content_hash: [u8; 32]) -> [u8; 256] {
    let mut result = [0; 256];
    result[..8].copy_from_slice(b"PBTQW25\0");
    for (offset, value) in [
        (0x08, 1),
        (0x0c, 256),
        (0x10, 0x01020304),
        (0x14, 7),
        (0x18, 291),
        (0x1c, 151936),
        (0x20, 151387),
        (0x24, promptboot_core::CONTEXT_LIMIT),
        (0x28, 24),
        (0x2c, 896),
        (0x30, 4864),
        (0x34, 14),
        (0x38, 2),
        (0x3c, 64),
        (0x40, 151645),
        (0x44, 151643),
        (0x48, 151643),
        (0x4c, 0),
    ] {
        put_u32(&mut result, offset, value);
    }
    for (offset, value) in [
        (0x50, gguf::SOURCE_BYTES),
        (0x58, file_size),
        (0x60, 256),
        (0x68, 448),
        (0x70, 5_947_744),
        (0x78, 422_782_464),
    ] {
        put_u64(&mut result, offset, value);
    }
    result[0x80..0xa0].copy_from_slice(&gguf::SOURCE_SHA256);
    result[0xa0..0xc0].copy_from_slice(&content_hash);
    result
}

fn section_entry(
    section_id: u32,
    element_type: u32,
    count: u64,
    offset: u64,
    length: u64,
    digest: [u8; 32],
) -> [u8; 64] {
    let mut result = [0; 64];
    put_u32(&mut result, 0, section_id);
    put_u32(&mut result, 4, element_type);
    put_u64(&mut result, 8, count);
    put_u64(&mut result, 16, offset);
    put_u64(&mut result, 24, length);
    result[32..64].copy_from_slice(&digest);
    result
}

fn write_padding(
    output: &mut File,
    target: u64,
    digest: Option<&mut Sha256>,
) -> Result<(), String> {
    let position = output
        .stream_position()
        .map_err(|error| format!("cannot query output position: {error}"))?;
    let length = target
        .checked_sub(position)
        .ok_or_else(|| "output layout moved backwards".to_owned())?;
    let zeros = vec![0; length as usize];
    output
        .write_all(&zeros)
        .map_err(|error| format!("cannot write padding: {error}"))?;
    if let Some(state) = digest {
        state.update(&zeros);
    }
    Ok(())
}

fn temporary_path(output: &Path) -> Result<PathBuf, String> {
    let name = output
        .file_name()
        .ok_or_else(|| "output must name a file".to_owned())?
        .to_string_lossy();
    Ok(output.with_file_name(format!(".{name}.{}.tmp", std::process::id())))
}

fn publish(temporary: &Path, output: &Path) -> Result<(), String> {
    fs::rename(temporary, output)
        .map_err(|error| format!("cannot publish packed model: {error}"))?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        File::open(parent)
            .and_then(|directory| directory.sync_all())
            .map_err(|error| format!("cannot sync output directory: {error}"))?;
    }
    Ok(())
}

fn pack(source_path: &Path, temporary: &Path, model: gguf::Gguf) -> Result<(), String> {
    let small = tokenizer_sections(&model.metadata)?;
    let mut offsets = Vec::with_capacity(7);
    let mut cursor = HEADER_BYTES + DIRECTORY_BYTES;
    for data in &small {
        cursor = align(cursor)?;
        offsets.push(cursor);
        cursor = cursor
            .checked_add(data.len() as u64)
            .ok_or_else(|| "section layout overflow".to_owned())?;
    }
    cursor = align(cursor)?;
    let tensor_directory_offset = cursor;
    offsets.push(cursor);
    cursor = cursor
        .checked_add(291 * 96)
        .ok_or_else(|| "tensor directory overflow".to_owned())?;
    cursor = align(cursor)?;
    let tensor_data_offset = cursor;
    offsets.push(cursor);

    struct Layout {
        source_offset: u64,
        destination: u64,
        length: u64,
    }
    let mut source =
        File::open(source_path).map_err(|error| format!("cannot open source: {error}"))?;
    let mut tensor_entries = Vec::with_capacity(291 * 96);
    let mut layouts = Vec::with_capacity(291);
    let mut data_cursor = tensor_data_offset;
    for (tensor_id, (name, dims, source_dtype, layer, role)) in
        gguf::pack_tensor_rows().into_iter().enumerate()
    {
        let tensor = model
            .tensors
            .get(&name)
            .ok_or_else(|| format!("source tensor disappeared: {name}"))?;
        data_cursor = align(data_cursor)?;
        let raw_hash = hash_region(&mut source, tensor.offset, tensor.length)?;
        let mut row = [0; 96];
        put_u32(&mut row, 0, tensor_id as u32);
        row[4..6].copy_from_slice(&layer.to_le_bytes());
        row[6..8].copy_from_slice(&role.to_le_bytes());
        put_u32(
            &mut row,
            8,
            match source_dtype {
                0 => 1,
                2 => 2,
                8 => 3,
                _ => return Err(format!("unmapped source dtype {source_dtype}")),
            },
        );
        put_u32(&mut row, 12, dims.len() as u32);
        for index in 0..4 {
            put_u32(
                &mut row,
                16 + index * 4,
                u32::try_from(*dims.get(index).unwrap_or(&1))
                    .map_err(|_| format!("tensor dimension exceeds u32: {name}"))?,
            );
        }
        put_u64(&mut row, 32, data_cursor);
        put_u64(&mut row, 40, tensor.length);
        row[48..80].copy_from_slice(&raw_hash);
        tensor_entries.extend(row);
        layouts.push(Layout {
            source_offset: tensor.offset,
            destination: data_cursor,
            length: tensor.length,
        });
        data_cursor = data_cursor
            .checked_add(tensor.length)
            .ok_or_else(|| "tensor data layout overflow".to_owned())?;
    }
    let file_size = data_cursor;
    if file_size > CEILING {
        return Err(format!("packed size {file_size} exceeds {CEILING}"));
    }

    let mut output = OpenOptions::new()
        .write(true)
        .read(true)
        .create_new(true)
        .open(temporary)
        .map_err(|error| format!("cannot create temporary output: {error}"))?;
    output
        .write_all(&vec![0; (HEADER_BYTES + DIRECTORY_BYTES) as usize])
        .map_err(|error| format!("cannot reserve header: {error}"))?;
    for (offset, data) in offsets[..5].iter().zip(&small) {
        write_padding(&mut output, *offset, None)?;
        output
            .write_all(data)
            .map_err(|error| format!("cannot write section: {error}"))?;
    }
    write_padding(&mut output, tensor_directory_offset, None)?;
    output
        .write_all(&tensor_entries)
        .map_err(|error| format!("cannot write tensor directory: {error}"))?;
    write_padding(&mut output, tensor_data_offset, None)?;
    let mut tensor_data_hash = Sha256::new();
    for layout in layouts {
        write_padding(&mut output, layout.destination, Some(&mut tensor_data_hash))?;
        copy_region(
            &mut source,
            &mut output,
            layout.source_offset,
            layout.length,
            &mut tensor_data_hash,
        )?;
    }
    if output
        .stream_position()
        .map_err(|error| format!("cannot query final output size: {error}"))?
        != file_size
    {
        return Err("final packed size mismatch".to_owned());
    }
    let definitions = [
        (1, 1, 151937, offsets[0], small[0].len() as u64),
        (2, 2, 1372758, offsets[1], small[1].len() as u64),
        (3, 2, 151936, offsets[2], small[2].len() as u64),
        (4, 3, 151387, offsets[3], small[3].len() as u64),
        (5, 2, 2509, offsets[4], small[4].len() as u64),
        (
            6,
            4,
            291,
            tensor_directory_offset,
            tensor_entries.len() as u64,
        ),
    ];
    let mut directory = Vec::with_capacity(DIRECTORY_BYTES as usize);
    for &(id, dtype, count, offset, length) in &definitions {
        let digest = hash_region(&mut output, offset, length)?;
        directory.extend(section_entry(id, dtype, count, offset, length, digest));
    }
    directory.extend(section_entry(
        7,
        2,
        file_size - tensor_data_offset,
        tensor_data_offset,
        file_size - tensor_data_offset,
        tensor_data_hash.finish(),
    ));
    output
        .seek(SeekFrom::Start(HEADER_BYTES))
        .and_then(|_| output.write_all(&directory))
        .map_err(|error| format!("cannot write section directory: {error}"))?;
    let content_hash = hash_region(&mut output, HEADER_BYTES, file_size - HEADER_BYTES)?;
    output
        .seek(SeekFrom::Start(0))
        .and_then(|_| output.write_all(&header(file_size, content_hash)))
        .map_err(|error| format!("cannot write header: {error}"))?;
    output
        .sync_all()
        .map_err(|error| format!("cannot sync packed model: {error}"))?;
    Ok(())
}

pub fn run(source: &Path, output: &Path) -> Result<InspectionReport, PackFailure> {
    gguf::verify_identity(source).map_err(PackFailure::Identity)?;
    let model = gguf::parse_path(source).map_err(PackFailure::Schema)?;
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            PackFailure::Pack(format!("cannot create output directory: {error}"))
        })?;
    }
    let temporary = temporary_path(output).map_err(PackFailure::Pack)?;
    if temporary.exists() {
        fs::remove_file(&temporary).map_err(|error| {
            PackFailure::Pack(format!("cannot remove stale temporary: {error}"))
        })?;
    }
    let result = (|| {
        pack(source, &temporary, model).map_err(PackFailure::Pack)?;
        let report = inspect::inspect(&temporary, source).map_err(PackFailure::Pack)?;
        publish(&temporary, output).map_err(PackFailure::Pack)?;
        Ok(report)
    })();
    if temporary.exists() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_failure_precedes_parsing_and_leaves_no_output() {
        let root =
            std::env::temp_dir().join(format!("promptboot-tools-pack-test-{}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let source = root.join("bad.gguf");
        let output = root.join("model.pbtqw25");
        fs::write(&source, b"GGUF-not-the-locked-asset").unwrap();
        assert!(matches!(
            run(&source, &output),
            Err(PackFailure::Identity(_))
        ));
        assert!(!output.exists());
        assert_eq!(fs::read_dir(&root).unwrap().count(), 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn publication_is_atomic_for_new_and_replaced_outputs() {
        let root = std::env::temp_dir().join(format!(
            "promptboot-tools-publish-test-{}",
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir(&root).unwrap();
        let output = root.join("model.pbtqw25");
        for existing in [None, Some(b"old".as_slice())] {
            if let Some(bytes) = existing {
                fs::write(&output, bytes).unwrap();
            }
            let temporary = root.join("packed.tmp");
            fs::write(&temporary, b"new").unwrap();
            publish(&temporary, &output).unwrap();
            assert_eq!(fs::read(&output).unwrap(), b"new");
            assert!(!temporary.exists());
        }
        fs::remove_dir_all(root).unwrap();
    }
}
