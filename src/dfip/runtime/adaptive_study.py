"""Native setup, traffic-only graph admission, frozen post-traffic admission."""

from dfip.runtime.serving_study import StudyWorker


class AdaptiveWorker(StudyWorker):
    def configure(self, mode):
        if mode not in ('native', 'adaptive'):
            raise ValueError('Pilot compares native and adaptive only')
        # Warm/check the native model without admitting fidelity-set shapes.
        report = super().configure('native')
        if mode == 'adaptive':
            self.engine.configure('adaptive')
            self.mode = mode
            report['mode'] = mode
            report['diagnostics'] = self.engine.diagnostics()
            report['initial_fidelity_scope'] = 'native setup before traffic-only adaptive admission'
        return report

    def finish(self):
        for graph in self.engine.adapter.graphs:
            if graph.admission is not None:
                graph.admission.frozen = True
        # Existing captures replay, but these checks cannot admit more shapes.
        return super().finish()
