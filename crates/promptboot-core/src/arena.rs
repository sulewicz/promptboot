use crate::PrimitiveStatus;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct Region {
    pub offset: u64,
    pub length: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ArenaUsage {
    pub capacity: u64,
    pub requested: u64,
    pub committed: u64,
    pub current: u64,
    pub high_water: u64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(C)]
pub struct ArenaError {
    pub status: PrimitiveStatus,
    pub needed_bytes: u64,
    pub available_bytes: u64,
}

impl ArenaError {
    const fn new(status: PrimitiveStatus, needed_bytes: u64, available_bytes: u64) -> Self {
        Self {
            status,
            needed_bytes,
            available_bytes,
        }
    }
}

pub struct Arena<'a> {
    storage: &'a mut [u8],
    requested: u64,
    committed: u64,
    current: u64,
    high_water: u64,
    sealed: bool,
}

impl<'a> Arena<'a> {
    pub fn new(storage: &'a mut [u8]) -> Result<Self, ArenaError> {
        if storage.is_empty() {
            return Err(ArenaError::new(PrimitiveStatus::ARENA_CAPACITY, 1, 0));
        }
        if (storage.as_mut_ptr() as usize) & 63 != 0 {
            return Err(ArenaError::new(PrimitiveStatus::ALIGNMENT, 64, 0));
        }
        Ok(Self {
            storage,
            requested: 0,
            committed: 0,
            current: 0,
            high_water: 0,
            sealed: false,
        })
    }

    pub fn allocate(&mut self, bytes: u64, alignment: u32) -> Result<Region, ArenaError> {
        if self.sealed {
            return Err(ArenaError::new(
                PrimitiveStatus::ARENA_SEALED,
                bytes,
                self.storage.len() as u64,
            ));
        }
        if bytes == 0 {
            return Err(ArenaError::new(PrimitiveStatus::LENGTH, 1, 0));
        }
        if alignment == 0 || !alignment.is_power_of_two() || alignment > 64 {
            return Err(ArenaError::new(
                PrimitiveStatus::ALIGNMENT,
                alignment as u64,
                64,
            ));
        }
        let mask = u64::from(alignment) - 1;
        let offset = self.committed.checked_add(mask).ok_or_else(|| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                bytes,
                self.storage.len() as u64,
            )
        })? & !mask;
        let end = offset.checked_add(bytes).ok_or_else(|| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                bytes,
                self.storage.len() as u64,
            )
        })?;
        let requested = self.requested.checked_add(bytes).ok_or_else(|| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                bytes,
                self.storage.len() as u64,
            )
        })?;
        let current = self.current.checked_add(bytes).ok_or_else(|| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                bytes,
                self.storage.len() as u64,
            )
        })?;
        if end > self.storage.len() as u64 {
            return Err(ArenaError::new(
                PrimitiveStatus::ARENA_CAPACITY,
                end,
                self.storage.len() as u64,
            ));
        }
        self.requested = requested;
        self.committed = end;
        self.current = current;
        if current > self.high_water {
            self.high_water = current;
        }
        Ok(Region {
            offset,
            length: bytes,
        })
    }

    pub fn seal(&mut self) -> Result<(), ArenaError> {
        if self.sealed {
            return Err(ArenaError::new(PrimitiveStatus::STATE, 0, 0));
        }
        self.sealed = true;
        Ok(())
    }

    pub fn reset(&mut self) -> Result<(), ArenaError> {
        if !self.sealed {
            return Err(ArenaError::new(PrimitiveStatus::STATE, 0, 0));
        }
        let committed = usize::try_from(self.committed).map_err(|_| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                self.committed,
                self.storage.len() as u64,
            )
        })?;
        unsafe { core::ptr::write_bytes(self.storage.as_mut_ptr(), 0, committed) };
        self.requested = 0;
        self.committed = 0;
        self.current = 0;
        self.sealed = false;
        Ok(())
    }

    pub fn usage(&self) -> ArenaUsage {
        ArenaUsage {
            capacity: self.storage.len() as u64,
            requested: self.requested,
            committed: self.committed,
            current: self.current,
            high_water: self.high_water,
        }
    }

    pub fn region(&self, region: Region) -> Result<&[u8], ArenaError> {
        if !self.sealed {
            return Err(ArenaError::new(PrimitiveStatus::STATE, 0, 0));
        }
        let (start, end) = self.region_bounds(region)?;
        Ok(unsafe { core::slice::from_raw_parts(self.storage.as_ptr().add(start), end - start) })
    }

    pub fn region_mut(&mut self, region: Region) -> Result<&mut [u8], ArenaError> {
        if !self.sealed {
            return Err(ArenaError::new(PrimitiveStatus::STATE, 0, 0));
        }
        let (start, end) = self.region_bounds(region)?;
        Ok(unsafe {
            core::slice::from_raw_parts_mut(self.storage.as_mut_ptr().add(start), end - start)
        })
    }

    fn region_bounds(&self, region: Region) -> Result<(usize, usize), ArenaError> {
        if region.length == 0 {
            return Err(ArenaError::new(PrimitiveStatus::LENGTH, 1, 0));
        }
        let end = region.offset.checked_add(region.length).ok_or_else(|| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                region.length,
                self.committed,
            )
        })?;
        if end > self.committed {
            return Err(ArenaError::new(
                PrimitiveStatus::LENGTH,
                end,
                self.committed,
            ));
        }
        let start = usize::try_from(region.offset).map_err(|_| {
            ArenaError::new(
                PrimitiveStatus::ARITHMETIC_OVERFLOW,
                region.offset,
                self.committed,
            )
        })?;
        let end = usize::try_from(end).map_err(|_| {
            ArenaError::new(PrimitiveStatus::ARITHMETIC_OVERFLOW, end, self.committed)
        })?;
        Ok((start, end))
    }
}
