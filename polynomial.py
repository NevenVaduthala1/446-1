#Polynomial class

import re
import numpy as np

class Polynomial:
    def __init__(self, coefficients=None, order=None):
        if coefficients is not None:
            self.coefficients = np.array(coefficients, dtype=int)
            self.order = len(self.coefficients) - 1
        elif order is not None:
            self.order = order
            self.coefficients = np.zeros(order + 1, dtype=int)

    def __repr__(self):
        return self.format_polynomial(self.coefficients)

    # Static method to create a Polynomial instance from a string
    @staticmethod
    def from_string(expression):
        coeff_array = Polynomial.extract_coefficients_ordered_by_power(expression)
        return Polynomial(coefficients=coeff_array)

    # Method to add two polynomials
    def __add__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = self.add_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    # Method to subtract two polynomials
    def __sub__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = self.subtract_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    # Method to multiply two polynomials
    def __mul__(self, other):
        if isinstance(other, Polynomial):
            result_coefficients = self.multiply_polynomials(self.coefficients, other.coefficients)
            return Polynomial(result_coefficients)

    # Method to divide two polynomials
    def __truediv__(self, other):
        if isinstance(other, Polynomial):
            # Ensure the denominator polynomial is not zero
            if np.all(other.coefficients == 0):
                raise ZeroDivisionError("Cannot divide by a zero polynomial.")
            # Return a RationalPolynomial where the current polynomial is the numerator and 'other' is the denominator
            return RationalPolynomial(self, other)

    # Method to check if two polynomials are equal
    def __eq__(self, other):
        if isinstance(other, Polynomial):
            return self.polynomials_are_equal(self.format_polynomial(self.coefficients), self.format_polynomial(other.coefficients))
        return False

    # Extract coefficients and powers from a polynomial string
    @staticmethod
    def extract_coefficients_ordered_by_power(expression):
        pattern = r"([+-]?\s*\d*)\*?x\^?(\d*)|([+-]?\s*\d+)(?!\*)"
        matches = re.findall(pattern, expression)
        highest_power = 0
        powers = []
        coefficients = []

        for match in matches:
            coefficient_str, power_str, constant_str = match

            if constant_str:  # This handles constants (e.g., +4 or -5)
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

    # Method to pad arrays to the same length (for addition/subtraction only)
    def pad_arrays(self, arr1, arr2):
        max_len = max(len(arr1), len(arr2))
        arr1_padded = np.pad(arr1, (max_len - len(arr1), 0), mode='constant')
        arr2_padded = np.pad(arr2, (max_len - len(arr2), 0), mode='constant')
        return arr1_padded, arr2_padded

    # Method to add two polynomials
    def add_polynomials(self, coeff1, coeff2):
        coeff1_padded, coeff2_padded = self.pad_arrays(coeff1, coeff2)
        return coeff1_padded + coeff2_padded

    # Method to subtract the first polynomial from the second
    def subtract_polynomials(self, coeff1, coeff2):
        coeff1_padded, coeff2_padded = self.pad_arrays(coeff1, coeff2)
        return coeff1_padded - coeff2_padded

    # Method to multiply two polynomials using convolution
    def multiply_polynomials(self, coeff1, coeff2):
        return np.convolve(coeff1, coeff2)

    # Method to format the polynomial
    def format_polynomial(self, coeff):
        ord = len(coeff)
        result = ""
        first_term = True  # Flag to track the first non-zero term

        # Check if all coefficients are zero
        if np.all(coeff == 0):
            return "0"

        for i in range(ord):
            coefficient = coeff[i]
            power = ord - (i + 1)

            # Skip if the coefficient is zero
            if coefficient == 0:
                continue

            # Handle different cases for the powers of x
            if power == 0:
                poly = f"{abs(coefficient)}"
            elif power == 1:
                poly = f"{abs(coefficient)}x"
            else:
                poly = f"{abs(coefficient)}x^{power}"

            # Handle the sign of the coefficient
            if first_term:  # For the first non-zero term, no leading sign if positive
                if coefficient < 0:
                    result += f"-{poly}"
                else:
                    result += f"{poly}"
                first_term = False
            else:  # For subsequent terms, include the sign
                if coefficient > 0:
                    result += f" + {poly}"
                else:
                    result += f" - {poly}"

        result = result.strip()
        result = result.replace("1x", "x").replace("-1x", "-x")
        return result

    # Method to check if two polynomials are equal
    def polynomials_are_equal(self, poly1, poly2):
        return poly1 == poly2

