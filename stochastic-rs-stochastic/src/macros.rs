//! Process backend switching, device handle and Python wrapper macros.

#[macro_use]
mod backend;

#[cfg(device_backend)]
#[macro_use]
mod device_handle;

#[macro_use]
mod dispatch;

#[macro_use]
mod process_methods;

#[macro_use]
mod process_1d;

#[macro_use]
mod process_2x1d;

#[macro_use]
mod process_2d;
