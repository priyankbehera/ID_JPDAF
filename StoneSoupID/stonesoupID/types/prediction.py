import datetime
import weakref
from collections.abc import Sequence
import numpy as np

from .array import CovarianceMatrix
from .base import Type
from .state import (State, GaussianState, EnsembleState,
                    ParticleState, MultiModelParticleState, RaoBlackwellisedParticleState,
                    SqrtGaussianState, InformationState, TaggedWeightedGaussianState,
                    WeightedGaussianState, CategoricalState, ASDGaussianState,
                    BernoulliParticleState, KernelParticleState, ASDTaggedWeightedGaussianState)
from ..base import Property
from ..models.transition.base import TransitionModel
from ..types.state import CreatableFromState, CompositeState, PointMassState


class Prediction(Type, CreatableFromState):
    """ Prediction type

    This is the base prediction class. """
    transition_model: TransitionModel = Property(
        default=None, doc='The transition model used to make the prediction')
    prior: State = Property(default=None)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self.prior, 'hypothesis'):
            if hasattr(self.prior.hypothesis, 'prediction'):
                prior_prediction_prior = getattr(self.prior.hypothesis.prediction, 'prior', None)
                if prior_prediction_prior is not None:
                    # Create weakref to avoid using significant memory
                    self.prior.hypothesis.prediction.prior = weakref.ref(prior_prediction_prior)

    @prior.getter
    def prior(self):
        if isinstance(self._property_prior, weakref.ReferenceType):
            return self._property_prior()
        else:
            return self._property_prior

    def __getstate__(self):
        state = super().__getstate__().copy()
        # Resolve weakref
        state['_property_prior'] = self.prior
        return state


class MeasurementPrediction(Type, CreatableFromState):
    """ Prediction type

    This is the base measurement prediction class. """


class StatePrediction(Prediction, State):
    """ StatePrediction type

    Most simple state prediction type, which only has time and a state vector.
    """


class InformationStatePrediction(Prediction, InformationState):
    """ InformationStatePrediction type

    Information state prediction type: contains state vector, precision matrix and timestamp
    """


class StateMeasurementPrediction(MeasurementPrediction, State):
    """ MeasurementPrediction type

    Most simple measurement prediction type, which only has time and a state
    vector.
    """


class GaussianStatePrediction(Prediction, GaussianState):
    """ GaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """


class ASDGaussianStatePrediction(Prediction, ASDGaussianState):
    """ ASDGaussianStatePrediction type

    This is a simple ASDGaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """
    act_timestamp: datetime.datetime = Property(
        doc="The timestamp for which the state is predicted")


class ASDTaggedWeightedGaussianStatePrediction(Prediction, ASDTaggedWeightedGaussianState):
    """ASD Tagged Weighted Gaussian Prediction"""
    act_timestamp: datetime.datetime = Property(
        doc="The timestamp for which the state is predicted")


class SqrtGaussianStatePrediction(Prediction, SqrtGaussianState):
    """ SqrtGaussianStatePrediction type

    This is a Gaussian state prediction object, with the covariance held
    as the square root of the covariance matrix
    """


class WeightedGaussianStatePrediction(Prediction, WeightedGaussianState):
    """ WeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution
    with an associated weight.
    """


class TaggedWeightedGaussianStatePrediction(Prediction,
                                            TaggedWeightedGaussianState):
    """ TaggedWeightedGaussianStatePrediction type

    This is a simple Gaussian state prediction object, which, as the name
    suggests, is described by a Gaussian distribution, with an associated
    weight and unique tag.
    """


class GaussianMeasurementPrediction(MeasurementPrediction, GaussianState):
    """ GaussianMeasurementPrediction type

    This is a simple Gaussian measurement prediction object, which, as the name
    suggests, is described by a Gaussian distribution.
    """

    cross_covar: CovarianceMatrix = Property(
        default=None, doc="The state-measurement cross covariance matrix")
    B: np.ndarray = Property(
        default=None, doc="ID‐form arc‐coefficient matrix for the measurement node")
    V: np.ndarray = Property(
        default=None, doc="ID‐form conditional variances for the measurement")

    def __init__(self, *args, **kwargs):
        B = kwargs.pop("B", None)
        V = kwargs.pop("V", None)
        super().__init__(*args, **kwargs)
        if self.cross_covar is not None \
                and self.cross_covar.shape[1] != self.state_vector.shape[0]:
            raise ValueError("cross_covar should have the same number of "
                             "columns as the number of rows in state_vector")\
        
        self.B = B
        self.V = V
    


# Don't need to support Sqrt Covar for MeasurementPrediction
CreatableFromState.class_mapping[MeasurementPrediction][SqrtGaussianState] = \
    GaussianMeasurementPrediction


class ASDGaussianMeasurementPrediction(MeasurementPrediction, ASDGaussianState):
    """ASD Gaussian Measurement Prediction"""
    cross_covar: CovarianceMatrix = Property(
        doc="The state-measurement cross covariance matrix", default=None)


class ASDTaggedWeightedGaussianMeasurementPrediction(
        MeasurementPrediction, ASDTaggedWeightedGaussianState):
    """ASD Tagged Weighted Gaussian Measurement Prediction"""
    cross_covar: CovarianceMatrix = Property(
        doc="The state-measurement cross covariance matrix", default=None)


class ParticleStatePrediction(Prediction, ParticleState):
    """ParticleStatePrediction type

    This is a simple Particle state prediction object.
    """


class PointMassStatePrediction(Prediction, PointMassState):
    """PointMassStatePrediction type

    This is a simple Point mass state prediction object.
    """


class ParticleMeasurementPrediction(MeasurementPrediction, ParticleState):
    """MeasurementStatePrediction type

    This is a simple Particle measurement prediction object.
    """


class PointMassMeasurementPrediction(MeasurementPrediction, PointMassState):
    """MeasurementStatePrediction type

    This is a simple Point mass measurement prediction object.
    """


class MultiModelParticleStatePrediction(Prediction, MultiModelParticleState):
    """MultiModelParticleStatePrediction type

    This is a simple multi-model Particle state prediction object.
    """


class RaoBlackwellisedParticleStatePrediction(Prediction, RaoBlackwellisedParticleState):
    """RaoBlackwellisedParticleStatePrediction type

    This is a simple Rao Blackwellised Particle state prediction object.
    """


class BernoulliParticleStatePrediction(Prediction, BernoulliParticleState):
    """BernoulliParticleStatePrediction type

    This is a simple Bernoulli Particle state prediction object"""


class KernelParticleStatePrediction(Prediction, KernelParticleState):
    """KernelParticleStatePrediction type

    This is a kernel particle state prediction object.
    """


class KernelParticleStateMeasurementPrediction(MeasurementPrediction, KernelParticleState):
    """KernelParticleStateMeasurementPrediction type

    This is a kernel particle state measurement prediction object.
    """


class EnsembleStatePrediction(Prediction, EnsembleState):
    """EnsembleStatePrediction type

    This is a simple Ensemble measurement prediction object.
    """


class EnsembleMeasurementPrediction(MeasurementPrediction, EnsembleState):
    """EnsembleMeasurementPrediction type

    This is a simple Ensemble measurement prediction object.
    """


class CategoricalStatePrediction(Prediction, CategoricalState):
    """Categorical state prediction type"""


class CategoricalMeasurementPrediction(MeasurementPrediction, CategoricalState):
    """Categorical measurement prediction type"""


class CompositePrediction(Prediction, CompositeState):
    """Composite prediction type

    Composition of :class:`~.Prediction`.
    """

    sub_states: Sequence[Prediction] = Property(
        doc="Sequence of sub-predictions comprising the composite prediction. All sub-predictions "
            "must have matching timestamp. Must not be empty.")


Prediction.register(CompositeState)  # noqa: E305


class CompositeMeasurementPrediction(MeasurementPrediction, CompositeState):
    """Composite measurement prediction type

    Composition of :class:`~.MeasurementPrediction`.
    """

    sub_states: Sequence[MeasurementPrediction] = Property(
        default=None,
        doc="Sequence of sub-measurement-predictions comprising the composite measurement "
            "prediction. All sub-measurement-predictions must have matching timestamp.")


MeasurementPrediction.register(CompositeState)  # noqa: E305
