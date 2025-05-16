"""
@author: Nicola Lombardi - HW Baseband - Telit Cinterion
SCRIPT     DOCUMENTATION  (RELEASE 1.0) - CLASSES FOR PERTURBATION TASK

Split the data to have 60% training samples and 40% test samples.
Create a package data_perturb that will contain three different classes of objects:
1. An abstract class CDataPerturb serving as an interface, which requires implementing an abstract method named data_perturbation,
   which takes as input a flat vector x and returns a perturbed version xp.

   Moreover, this class should implement also a (concrete) method named perturb_dataset(X) which will iteratively apply data_perturbation
   to each row of the dataset X and return the perturbed version of the whole dataset Xp.

2. A child class inherited from CDataPerturb, named 'CDataPerturbRandom', which randomly changes K values in the input vector x,
   selecting such values uniformly in the range [min_value, max_value].
   The constructor of the class will take as input parameters: min_value, max_value, and K, with default
   values respectively of 0, 255, 10. For all these parameters, setters and getters should be available.

3. Another child class inherited from CDataPerturb, named 'CDataPerturbGaussian', which randomly perturbs all values in the input vector x with Gaussian noise.
   The Gaussian noise must have zero mean and standard deviation parametrized by sigma.

   Hint: use sigma * np.random.randn(...) to rescale the values sampled
   from the standard normal with zero mean and unit variance. If the values in the perturbed image are below min_value or above max_value,
   they should be set to min_value and max_value respectively.

   The constructor will take min_value, max_value and sigma as input parameters, having default values of 0, 255, and 100.0.
   Setters/getters should be available for all these parameters. Test both perturbation models on ten random images drawn from the MNIST dataset,
   and visually compare the results. Hint: you can use the function plot_ten_images(). Train the NMC classifier on the training data and test it on the test set.
   Compute the classification accuracy, that is, the fraction of correctly classified samples in the test set. Perturb the digit images in the test set using
   the two perturbation models with the following parameter values:

   K=[0, 10, 20, 50, 100, 200, 500] and sigma=[10, 20, 200, 200, 500]. Compute the classification accuracy values against K.
   Compute the classification accuracy values against sigma. Create a plot with two subplots, plot accuracy vs K in the leftmost plot, and accuracy vs sigma in
   the rightmost plot
"""
import random
from abc import ABC, abstractmethod
import utils as u
import numpy as np

# ============================================================================
# -----------------------------------
#           interface CLASS
# -----------------------------------
# ============================================================================

"""
In object-oriented programming, an abstract class is a class that cannot be instantiated. 
However, it can be created classes that inherit from an abstract class.

Typically, it can be used an abstract class to create a blueprint for other classes.

Similarly, an abstract method is an method without an implementation. 
"""

# interface (generic strategy) - ABSTRACT CLASS
class CDataPerturb(ABC):
    """
    An abstract class 'CDataPerturb' serving as an interface, which requires implementing an abstract
    method named data_perturbation, which takes as input a flat vector "X" and returns a perturbed
    version Xp. Moreover, this class should implement also a (concrete) method named perturb_dataset(X)
    which will iteratively apply data_perturbation to each row of the dataset X and
    return the perturbed version of the whole dataset "Xp"
    """

    # concrete method
    def __str__(self):
        """ Example
        obj_attributes_dict = self.__dict__
        obj_attributes_string = u.describe_obj(obj_attributes_dict)

		:return: Java style object image (toString method). This method print every attribute of an object
		"""

        return "   (__STR__ method)\n'"+ self.__class__.__name__+ "' " + "\n" + "\nAttributes Object:\n" + u.describe_obj(self.__dict__)


    @abstractmethod
    def data_perturbation(self, flat_x_vector):  # according used strategy (strategy pattern)
        """ to extend according to chosen probability distribution [ BEHAVIOUR COULD CHANGE ]
        :param flat_x_vector: input (a row of dataframe)
        :return: output (perturbed version of flat_x_vector)
        """
        pass

    # concrete method
    def perturb_dataset(self, dataframe):
        """
        It will iteratively apply data_perturbation() to each row of the dataset X and
        return the perturbed version of the whole dataset Xp. This will be independent
        by Probability Distribution

        -Instead of:

        for row in dataframe:
            #print("")
            # row = self.data_perturbation(flat_x_vector = row)

        -Use:

        with comprehension : list_variable = [x for x in iterable]

        :param dataframe: [X1, X2, ..., XN]
        :return: Xp
        """
        return np.array(self.data_perturbation(flat_x_vector=row) for row in dataframe)

# ============================================================================
# -----------------------------------
#           Random Perturbation
# -----------------------------------
# ============================================================================
class CDataPerturbRandom(CDataPerturb):
    def __init__(self, min_value=0, max_value=255, k=10):
        self.min_value = min_value
        self.max_value = max_value
        self.k = k

    def get_params(self): # avoid six methods definition (three getters and three setters)
        """ :return: object attribute values """
        return self.k, self.min_value, self.max_value
    def data_perturbation(self, flat_x_vector):
        """
        :used probability distribution:  p(n) = 1 / [ b - a +1 ] = 1 / [ max_value - min_value + 1] (probability mass function)

        :scope of method:
        A child class inherited from CDataPerturb, named CDataPerturbRandom, which randomly changes K values in the input vector x,
        selecting such values uniformly in the range [min_value, max_value].

        Assignment statements in Python do not copy objects, they create bindings between a target and an object.
        For collections that are mutable or contain mutable items, a copy is sometimes needed so one can change one copy without changing the other.

        Used numpy functions:
            numpy.random.uniform(Lower boundary of the output interval, Upper boundary of the output interval, Output shape)
            numpy.random.choice(vector, size= Output shape, replace=True, p= The probabilities associated with each entry in a)

        Example : [X] -----------> [Xp]

        X = [68, 15, 55, 0, 53, 29, 30, 83, 15, 21]                  # len -> 10
        Xs = list(np.random.choice(len(X), 10, replace=False))       # list( array([3, 8, 7, 4, 0, 9, 1, 2, 5, 6]) )
        Xp = X.copy()                                                # [68, 15, 55, 0, 53, 29, 30, 83, 15, 21]
        for el in Xs:
            Xp[el] = float( X[el] + np.random.uniform(0, -100, 1)  )

        Xp =
                    [28.81640489400889,
                     -80.54026054285669,
                     47.68436446861046,
                     -4.783320221419684,
                     -4.813028663461807,
                     -5.768412134538131,
                     -51.350239683546874,
                     39.65520548063099,
                     -22.90377952398577,
                     -43.74288721561318]

        :param flat_x_vector: [X], population in which it will be chosen a "vial" of samples
        :return: [Xp]
        """
        perturbed_vector = flat_x_vector.copy()  # shallow copy constructs a new object and then inserts references into it to the objects found in the original.
        indices_to_perturb = list(np.random.choice(len(flat_x_vector), self.k, replace=False) ) # Generates a random sample from a given 1-D array
        for el in indices_to_perturb:
            perturbed_vector[el] = float( flat_x_vector[el] + np.random.uniform(self.min_value, self.max_value, 1) ) # output shape must be "1" for each iteration
        return perturbed_vector

# ============================================================================
# -----------------------------------
#          Gaussian Perturbation
# -----------------------------------
# ============================================================================

class CDataPerturbGaussian(CDataPerturb):
    def __init__(self, min_value=0, max_value=255, sigma=100.0):
        self.min_value = min_value
        self.max_value = max_value
        self.sigma = sigma

    def get_params(self): # avoid six methods definition (three getters and three setters)
        """ :return: object attribute values """
        return self.sigma, self.min_value, self.max_value

    def data_perturbation(self, flat_x_vector):
        """ Another child class inherited from CDataPerturb, named CDataPerturbGaussian, which randomly perturbs all values in the input vector x with Gaussian noise.
         :used probability distribution:  p(n) = 1 / [ b - a +1 ] = 1 / [ max_value - min_value + 1] (probability mass function)

        :scope of method:
        A child class inherited from CDataPerturb, named CDataPerturbGaussian, which randomly changes a sample of an interval [μ ± z * σ] built with a mean
        that is the sample of the input vector 'x', where 'z' is the random value.

        Used numpy functions:
            numpy.random.randn(value) -> Array of defined shape, filled with random floating-point samples from the standard normal distribution.
            numpy.clip(vector_to_clip, min, max) -> integer or float band pass filter

        Example : [X] -----------> [Xp]

        X = [68, 15, 55, 0, 53, 29, 30, 83, 15, 21]                  # len -> 10
        Xs = list(np.random.choice(len(X), 10, replace=False))       # list( array([3, 8, 7, 4, 0, 9, 1, 2, 5, 6]) )
        Xp = X.copy()                                                # [68, 15, 55, 0, 53, 29, 30, 83, 15, 21]
        for el in Xs:
            Xp[el] = float( X[el] + np.random.uniform(0, -100, 1)  )

        Xp =
                    [28.81640489400889,
                     -80.54026054285669,
                     47.68436446861046,
                     -4.783320221419684,
                     -4.813028663461807,
                     -5.768412134538131,
                     -51.350239683546874,
                     39.65520548063099,
                     -22.90377952398577,
                     -43.74288721561318]


        :param flat_x_vector:
        :return: [Xp]
        """
        perturbed_vector = flat_x_vector.copy()
        for idx in range(len(flat_x_vector)):
            perturbed_vector[idx] = flat_x_vector[idx] + self.sigma * float(np.random.randn(1)) # random value (only one) fetched from interval [μ ± kσ] where μ = X[idx]
        perturbed_vector = list(np.clip(perturbed_vector, self.min_value, self.max_value))
        return perturbed_vector

# ============================================================================
# -----------------------------------
#          Test Perturbation section
# -----------------------------------
# ============================================================================
def test_data_uniform_perturbation(disruptor):
    """
    :param x: flat vector [X]
    :param disruptor: CDataPerturbRandom(min_value, max_value, k)  transforms '[X]' in '[Xp]'
    :return: print [Xp]



    """
    params = disruptor.get_params()  # k, min_value, max_value
    false_flat_vector = [random.randint(params[1], params[-1]) for _ in range(params[0])]  # vector to test filled by 10 random integers between 100 and 0
    print(f"\nInput X : {false_flat_vector}\n\n")
    print(f"The disruptor used is an Instance of -> {disruptor}")  # it calls __str__() method
    print("Processing : [X] ---> [Xp] \n")
    Xp_unif = disruptor.data_perturbation(flat_x_vector=false_flat_vector)
    print(Xp_unif)
    print("TEST PASS ? len([X]) == len([Xp]) ")
    u.display_status_msg(test_pass_y_or_n(x=false_flat_vector, y=Xp_unif))

def test_data_gaussian_perturbation(disruptor):
    """
    :param x: flat vector [X]
    :param disruptor: CDataPerturbGaussian(min_value, max_value, σ)  transforms '[X]' in '[Xp]'
    :return: print [Xp]
    """
    params = disruptor.get_params()  # k, min_value, max_value
    false_flat_vector = [random.randint(params[1], params[-1]) for _ in range(int(params[0]/10))]  # vector to test filled by 10 random integers between 100 and 0
    print(f"\nInput X : {false_flat_vector}\n\n")
    print(f"The disruptor used is an Instance of -> {disruptor}")  # it calls __str__() method
    print("Processing : [X] ---> [Xp] \n")
    Xp_gauss = disruptor.data_perturbation(flat_x_vector=false_flat_vector)
    print(Xp_gauss)
    print("TEST PASS ? len([X]) == len([Xp]) ")
    u.display_status_msg(test_pass_y_or_n(x=false_flat_vector, y=Xp_gauss))

def test_pass_y_or_n(x,y):
    return len(x) == len(y)


if __name__ == '__main__':
    # debug : PASS
    d1 = CDataPerturbRandom() # first disruptor keeping default init() argument values
    test_data_uniform_perturbation(d1) # strategy : Uniform
    #debug : PASS
    d2 = CDataPerturbGaussian() # second disruptor keeping default init() argument values
    test_data_gaussian_perturbation(d2) # strategy : Gaussian
