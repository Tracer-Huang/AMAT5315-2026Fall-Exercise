//! Physics shared by the executable, experiments, and tests.

pub mod dynamics;
pub mod fluid;
pub mod pair;
pub mod periodic;

pub fn greeting() -> &'static str {
    "Hello, world!"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greeting_is_hello_world() {
        assert_eq!(greeting(), "Hello, world!");
    }
}
