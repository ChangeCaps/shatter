pub trait EntryPointInput {
    type State;

    fn init() -> Self::State;

    fn run(self, state: &mut Self::State);
}

pub struct EntryPoint<In>
where
    In: EntryPointInput,
{
    state: In::State,
}

impl<In> EntryPoint<In>
where
    In: EntryPointInput,
{
    #[inline]
    pub fn new() -> Self {
        Self { state: In::init() }
    }

    #[inline]
    pub fn run(&mut self, input: In) {
        input.run(&mut self.state);
    }
}
