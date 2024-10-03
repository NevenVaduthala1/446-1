#Polynomial class

import re
import numpy as np
import sympy as sp

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

class RationalPolynomial:
    def __init__(self, numerator: Polynomial, denominator: Polynomial):
        self.numerator = numerator
        self.denominator = denominator
        if np.all(self.denominator.coefficients == 0):
            raise ValueError("The denominator cannot be zero.")
        
        self.simplify()  # Simplify the rational polynomial upon initialization

    def __repr__(self):
        # Check if the denominator is effectively 1
        if self.denominator == "1" or (len(self.denominator.coefficients) == 1 and self.denominator.coefficients[0] == 1):
            return f"({self.numerator})"
        return f"({self.numerator})/({self.denominator})"

    # Static method to parse a rational polynomial from a string
    @staticmethod
    def from_string(expression):
        # Clean and split the expression into numerator and denominator
        expression = expression.replace(' ', '')  # Remove spaces
        if '/' not in expression:
            raise ValueError("Expression does not represent a rational polynomial")
        
        numerator_str, denominator_str = expression.split('/')
        
        # Create Polynomial objects for both parts
        numerator = Polynomial.from_string(numerator_str)
        denominator = Polynomial.from_string(denominator_str)
        
        return RationalPolynomial(numerator, denominator)

    # Method to simplify the rational polynomial by checking for common factors
    def simplify(self):
        num_expr = self.to_sympy_expr(self.numerator)
        denom_expr = self.to_sympy_expr(self.denominator)

        # Find the GCD of the numerator and denominator
        gcd_expr = sp.gcd(num_expr, denom_expr)

        if gcd_expr != 1:
            # Simplify the numerator and denominator by dividing by the GCD
            simplified_numerator = sp.simplify(num_expr / gcd_expr)
            simplified_denominator = sp.simplify(denom_expr / gcd_expr)

            # Rebuild the polynomial representations from the simplified SymPy expressions
            self.numerator = self.from_sympy_expr(simplified_numerator)
            self.denominator = self.from_sympy_expr(simplified_denominator)

    # Convert a Polynomial object to a Sympy expression
    @staticmethod
    def to_sympy_expr(poly):
        x = sp.symbols('x')
        coeffs = poly.coefficients
        expr = sum(coeff * x**power for power, coeff in enumerate(reversed(coeffs)))
        return expr

    # Convert a Sympy expression back to a Polynomial object
    @staticmethod
    def from_sympy_expr(expr):
        x = sp.symbols('x')
        try:
            # Attempt to get the polynomial form
            poly_expr = sp.Poly(expr, x)
            coeffs = poly_expr.all_coeffs()
            coeffs = [int(c) for c in coeffs]  # Convert sympy integers to Python integers
            return Polynomial(coefficients=coeffs)
        except sp.PolynomialError:
            # If the result is not a polynomial (like 1/(x - 2)), return the expression directly
            return str(expr)

    # Addition of two rational polynomials
    def __add__(self, other):
        if isinstance(other, RationalPolynomial):
            # a/b + c/d = (a*d + b*c) / (b*d)
            new_numerator = (self.numerator * other.denominator) + (other.numerator * self.denominator)
            new_denominator = self.denominator * other.denominator

            # Display the intermediate addition step
            print(f"Addition (Before Simplification): ({self.numerator}) * ({other.denominator}) + ({other.numerator}) * ({self.denominator})")
            print(f"= {new_numerator}/{new_denominator}")
            
            # Return the resulting rational polynomial
            result = RationalPolynomial(new_numerator, new_denominator)
            result.simplify()  # Simplify the result
            
            # Display the simplified result
            print(f"Simplified Result: {result}")
            return result

    # Subtraction of two rational polynomials
    def __sub__(self, other):
        if isinstance(other, RationalPolynomial):
            # a/b - c/d = (a*d - b*c) / (b*d)
            new_numerator = (self.numerator * other.denominator) - (other.numerator * self.denominator)
            new_denominator = self.denominator * other.denominator
            
            # Display the intermediate subtraction step
            print(f"Subtraction (Before Simplification): ({self.numerator}) * ({other.denominator}) - ({other.numerator}) * ({self.denominator})")
            print(f"= {new_numerator}/{new_denominator}")
            
            # Return the resulting rational polynomial
            result = RationalPolynomial(new_numerator, new_denominator)
            result.simplify()  # Simplify the result
            
            # Display the simplified result
            print(f"Simplified Result: {result}")
            return result

    # Multiplication of two rational polynomials
    def __mul__(self, other):
        if isinstance(other, RationalPolynomial):
            # a/b * c/d = (a * c) / (b * d)
            new_numerator = self.numerator * other.numerator
            new_denominator = self.denominator * other.denominator

            # Display the intermediate multiplication step
            print(f"Multiplication (Before Simplification): ({self.numerator}) * ({other.numerator})")
            print(f"= {new_numerator}/{new_denominator}")
            
            # Return the resulting rational polynomial
            result = RationalPolynomial(new_numerator, new_denominator)
            result.simplify()  # Simplify the result
            
            # Display the simplified result
            print(f"Simplified Result: {result}")
            return result

    # Division of two rational polynomials
    def __truediv__(self, other):
        if isinstance(other, RationalPolynomial):
            # a/b ÷ c/d = (a * d) / (b * c)
            new_numerator = self.numerator * other.denominator
            new_denominator = self.denominator * other.numerator

            # Display the intermediate division step
            print(f"Division (Before Simplification): ({self.numerator}) * ({other.denominator}) / ({self.denominator}) * ({other.numerator})")
            print(f"= {new_numerator}/{new_denominator}")
            
            # Return the resulting rational polynomial
            result = RationalPolynomial(new_numerator, new_denominator)
            result.simplify()  # Simplify the result
            
            # Display the simplified result
            print(f"Simplified Result: {result}")
            return result

    # Equality check of two rational polynomials
    def __eq__(self, other):
        if isinstance(other, RationalPolynomial):
            # Cross-multiply to check for equality
            left = self.numerator * other.denominator
            right = self.denominator * other.numerator
            return left == right
        return False
