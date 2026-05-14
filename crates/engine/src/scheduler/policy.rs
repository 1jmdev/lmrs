/// Ordering policy used by the scheduler.
///
/// # Example
///
/// ```
/// use engine::SchedulePolicy;
///
/// assert_eq!(SchedulePolicy::default(), SchedulePolicy::Fcfs);
/// ```
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SchedulePolicy {
    /// First-come first-served, preserving insertion order.
    #[default]
    Fcfs,
    /// Higher priority groups run before lower priority groups.
    Priority,
}
