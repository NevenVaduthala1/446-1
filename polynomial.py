import numpy as np
import re 

class Polynomial:
    def __init__(self, coefficients=None, order=None):
        if coefficients is not None:
            self.coefficients = np.array(coefficients, dtype=int)
            self.order = len(self.coefficients) - 1
        elif order is not None:
            self.order = order
            self.coefficients = np.zeros(order + 1, dtype=int)

    def __repr__(self):
        return format_polynomial(self.coefficients)

    @staticmethod
    def from_string(expression):
        coeff_array = Polynomial.extract_coefficients_ordered_by_power(expression)
        return Polynomial(coefficients=coeff_array)

    def __add__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = add_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    def __sub__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = subtract_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    def __mul__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = multiply_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return polynomials_are_equal(format_polynomial(self.coefficients), format_polynomial(other.coefficients))
        return False

    # Division method (this is where __truediv__ goes)
    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            if np.all(other.coefficients == 0):
                raise ZeroDivisionError("Cannot divide by a zero polynomial.")
            return RationalPolynomial(self, other)
        else:
            raise TypeError("Can only divide by another Polynomial.")

    @staticmethod
    def extract_coefficients_ordered_by_power(expression):
        pattern = r"([+-]?\s*\d*)\*?x\^?(\d*)|([+-]?\s*\d+)(?!\*)"
        matches = re.findall(pattern, expression)
        highest_power = 0
        powers = []
        coefficients = []

        for match in matches:
            coefficient_str, power_str, constant_str = match

            if constant_str:
                coefficient = int(constant_str.replace(" ", ""))
                power = 0
            else:
                coefficient_str = coefficient_str.replace(" ", "")
                if coefficient_str == '' or coefficient_str == '+':
                    coefficient = 1
                elif coefficient_str == '-':
                    coefficient = -1
                else:
                    coefficient = int(coefficient_str)

                if power_str == '':
                    power = 1
                else:
                    power = int(power_str)

            coefficients.append(coefficient)
            powers.append(power)
            highest_power = max(highest_power, power)

        ordered_coefficients = [0] * (highest_power + 1)
        for coefficient, power in zip(coefficients, powers):
            ordered_coefficients[highest_power - power] = coefficient

        coefficients_array = np.array(ordered_coefficients)
        return coefficients_array

