import numpy as np

from .base import Attack
from .base import call_decorator
from .. import nprng


class GaussianNoiseAttack(Attack):
    """Increases the amount of salt and pepper noise until the input is misclassified.

    """

    @call_decorator
    def __call__(self, input_or_adv, label=None, unpack=True,
                 epsilons=100, repetitions=10):

        """Increases the amount of salt and pepper noise until the input is misclassified.

        Parameters
        ----------
        input_or_adv : `numpy.ndarray` or :class:`Adversarial`
            The original, unperturbed input as a `numpy.ndarray` or
            an :class:`Adversarial` instance.
        label : int
            The reference label of the original input. Must be passed
            if `a` is a `numpy.ndarray`, must not be passed if `a` is
            an :class:`Adversarial` instance.
        unpack : bool
            If true, returns the adversarial input, otherwise returns
            the Adversarial object.
        epsilons : int
            Number of steps to try between probability 0 and 1.
        repetitions : int
            Specifies how often the attack will be repeated.

        """

        a = input_or_adv
        del input_or_adv
        del label
        del unpack

        x = a.unperturbed
        min_, max_ = a.bounds()
        axis = a.channel_axis(batch=False)
        channels = x.shape[axis]
        shape = list(x.shape)
        shape[axis] = 1
        r = max_ - min_
        pixels = np.prod(shape)

        epsilons = min(epsilons, pixels)
        max_epsilon = 1

        for _ in range(repetitions):
            for epsilon in np.linspace(0, max_epsilon, num=epsilons + 1)[1:]:
                sigma = epsilon * r  # Standard deviation scaling with epsilon

                # Generate Gaussian noise
                noise = np.random.normal(0, sigma, x.shape).astype(x.dtype)
                
                # Add noise to the original image
                perturbed = x + noise
                perturbed = np.clip(perturbed, min_, max_)

                # Check for adversarial effectiveness
                if a.normalized_distance(perturbed) >= a.distance:
                    continue

                _, is_adversarial = a.forward_one(perturbed)
                if is_adversarial:
                    # Adjust epsilon based on effectiveness
                    max_epsilon = min(1, epsilon * 1.2)
                    break
