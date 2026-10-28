use super::*;

pub(crate) fn is_private_memory_database(database: &Path) -> bool {
    database.as_os_str().is_empty() || database == Path::new(":memory:")
}

pub(crate) fn normalized_authority_path_key(database: &Path) -> String {
    let database_string = database.to_string_lossy().into_owned();
    #[cfg(windows)]
    {
        database_string
            .strip_prefix(r"\\?\")
            .unwrap_or(&database_string)
            .to_ascii_lowercase()
    }
    #[cfg(not(windows))]
    {
        database_string
    }
}

pub(crate) fn file_identity_authority_key(database: &Path) -> SqliteStoreResult<Option<String>> {
    let _metadata = match fs::metadata(database) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    #[cfg(windows)]
    {
        use std::os::windows::ffi::OsStrExt;
        use winapi::shared::minwindef::FALSE;
        use winapi::um::fileapi::{
            BY_HANDLE_FILE_INFORMATION, CreateFileW, GetFileInformationByHandle, OPEN_EXISTING,
        };
        use winapi::um::handleapi::{CloseHandle, INVALID_HANDLE_VALUE};
        use winapi::um::winbase::FILE_FLAG_BACKUP_SEMANTICS;
        use winapi::um::winnt::{
            FILE_READ_ATTRIBUTES, FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_SHARE_WRITE,
        };

        let path = database
            .as_os_str()
            .encode_wide()
            .chain(std::iter::once(0))
            .collect::<Vec<_>>();
        let handle = unsafe {
            CreateFileW(
                path.as_ptr(),
                FILE_READ_ATTRIBUTES,
                FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                std::ptr::null_mut(),
                OPEN_EXISTING,
                FILE_FLAG_BACKUP_SEMANTICS,
                std::ptr::null_mut(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(std::io::Error::last_os_error().into());
        }
        let mut information: BY_HANDLE_FILE_INFORMATION = unsafe { std::mem::zeroed() };
        let result = unsafe { GetFileInformationByHandle(handle, &mut information) };
        let error = if result == FALSE {
            Some(std::io::Error::last_os_error())
        } else {
            None
        };
        unsafe {
            CloseHandle(handle);
        }
        if let Some(error) = error {
            return Err(error.into());
        }
        let file_index =
            (u64::from(information.nFileIndexHigh) << 32) | u64::from(information.nFileIndexLow);
        Ok(Some(format!(
            "file-id:{:x}:{file_index:x}",
            information.dwVolumeSerialNumber
        )))
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        Ok(Some(format!(
            "file-id:{:x}:{:x}",
            _metadata.dev(),
            _metadata.ino()
        )))
    }
    #[cfg(not(any(windows, unix)))]
    {
        let _ = _metadata;
        Ok(None)
    }
}

pub(crate) fn normalized_database_path(database: &Path) -> PathBuf {
    let absolute = if database.is_absolute() {
        database.to_path_buf()
    } else {
        std::env::current_dir()
            .map(|directory| directory.join(database))
            .unwrap_or_else(|_| database.to_path_buf())
    };
    if let Ok(canonical) = fs::canonicalize(&absolute) {
        return canonical;
    }
    let Some(file_name) = absolute.file_name() else {
        return absolute;
    };
    absolute
        .parent()
        .and_then(|parent| fs::canonicalize(parent).ok())
        .map(|parent| parent.join(file_name))
        .unwrap_or(absolute)
}
