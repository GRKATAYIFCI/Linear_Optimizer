import numpy as np
from loguru import logger


file_path = 'input/constraints.txt'


class LinearOpt:
    def __init__(self):
        self.objective_function = []
        self.A_matrix = []
        self.sign = []
        self.rhs = []
        self.bound = []
        self.basis = []
        self.B_inv = []
        self.c_n = []
        self.c_b = []
        self.current_value = 0

    def adjust_for_slack_variables(self):
        num_constraints = len(self.sign)
        num_original_vars = len(self.objective_function)

        # Add slack variables for 'L' constraints
        for index, sign in enumerate(self.sign):
            if sign == 'L':
                self.objective_function.append(0)  # Coefficient of slack variable in the objective function
                self.bound.append((len(self.objective_function) - 1, 0, 'i'))  # Bounds for slack variables

                # Add slack variable column to A matrix
                for j in range(num_constraints):
                    if j == index:
                        self.A_matrix[j].append(1.0)  # Add 1 to the corresponding constraint
                    else:
                        self.A_matrix[j].append(0.0)  # Add 0 to other constraints
        #return objective_function, A_matrix, bounds

    def read_constraints_from_file(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()

            objective_function = [float(num) for num in lines[0].strip().split()]

            A_matrix = []
            i = 1
            while lines[i].strip()[0].isdigit() or lines[i].strip()[0] == '-':
                A_matrix.append([float(num) for num in lines[i].strip().split()])
                i += 1

            signs = list(lines[i].strip())  # Each character is treated as an individual sign
            i += 1

            rhs = [float(num) for num in lines[i].strip().split()]
            i += 1

            bounds = []
            for line in lines[i:]:
                bounds.append(tuple(line.strip().strip('()').split(',')))

            self.objective_function = objective_function
            self.A_matrix = A_matrix
            self.sign = signs
            self.rhs = rhs
            self.bound = bounds
            return objective_function, A_matrix, signs, rhs, bounds

    def transform_unbounded_variables(self):
        num_variables = len(self.bound)
        extended_A_matrix = [list(row) for row in self.A_matrix]  # Copy the A_matrix
        extended_objective = list(self.objective_function)  # Copy the objective function
        extended_bounds = []

        for i in range(num_variables):
            variable_index, lower_bound, upper_bound = self.bound[i]
            if lower_bound == -np.inf:

                extended_objective.append(self.objective_function[i])
                extended_objective.append(-self.objective_function[i])

                for row in extended_A_matrix:
                    row.append(row[i])
                    row.append(-row[i])

                extended_bounds.append((variable_index, 0, np.inf))
                extended_bounds.append((variable_index, 0, np.inf))
            else:
                extended_bounds.append((variable_index, lower_bound, upper_bound))

        self.A_matrix = extended_A_matrix
        self.objective_function = extended_objective
        self.bound = extended_bounds

    def adjust_bounds(self):
        new_bounds = []
        num_variables = len(self.bound)

        for i in range(num_variables):
            variable_index, lower_bound, upper_bound = self.bound[i]
            variable_index = int(variable_index)
            if lower_bound == '-i':
                lower_bound = -np.inf
            else:
                lower_bound = float(lower_bound)

            if upper_bound == 'i':
                upper_bound = np.inf
            else:
                upper_bound = float(upper_bound)
            new_bounds.append((variable_index, lower_bound, upper_bound))
        self.bound = new_bounds

        adjusted_A_matrix = [row[:] for row in self.A_matrix]  # Deep copy the A matrix
        adjusted_b_vector = self.rhs[:]
        adjusted_bounds = []
        for i in range(num_variables):
            variable_index, lower_bound, upper_bound = self.bound[i]

            if lower_bound == -np.inf:
                self.transform_unbounded_variables()

            if lower_bound != -np.inf and lower_bound != 0:
                # Adjust all constraints that involve this variable
                for constraint_index in range(len(self.A_matrix)):
                    adjusted_b_vector[constraint_index] -= adjusted_A_matrix[constraint_index][i] * lower_bound

            new_lower_bound = 0
            new_upper_bound = upper_bound - lower_bound if upper_bound != np.inf else np.inf

            if upper_bound != np.inf:
                new_constraint = [0] * num_variables
                new_constraint[i] = 1
                adjusted_A_matrix.append(new_constraint)
                adjusted_b_vector.append(upper_bound - lower_bound)
                new_upper_bound = np.inf

            adjusted_bounds.append((i, new_lower_bound, new_upper_bound))

        self.bound = adjusted_bounds
        self.A_matrix = adjusted_A_matrix
        self.rhs = adjusted_b_vector

    def current_revised_simplex_elements(self):
        [r, c] = self.A_matrix.shape
        self.basis = list(range(c - r, c))
        self.B_inv = np.linalg.inv(self.A_matrix[:, self.basis])
        while True:
            x_b = self.B_inv.dot(self.rhs)
            x = np.zeros(c)
            x[self.basis] = x_b
            self._print_current_values(x)
            self.c_b = self.objective_function[self.basis]
            self.c_n = self.objective_function[[j for j in range(c) if j not in self.basis]]
            self.check_vector = self.c_b.dot(self.B_inv).dot(
                self.A_matrix[:, [j for j in range(c) if j not in self.basis]]) - self.c_n

            logger.debug(self.objective_function.dot(x))
            logger.debug(self.check_vector)
            if np.all(self.check_vector >= 0):
                logger.error("SOLUTION FOUND")
                return x  # Optimal solution found

            # Determine entering variable (smallest index of negative reduced cost)
            entering = np.argmin(self.check_vector)

            # Determine leaving variable
            d = self.B_inv.dot(self.A_matrix[:, entering])
            ratios = np.array([x_b[i] / d[i] if d[i] > 0 else np.inf for i in range(len(d))])
            leaving = np.argmin(ratios)
            # Update basis
            # Update inverse of the basis matrix B_inv using Sherman-Morrison or similar method
            logger.warning(f"leaving variable is {leaving}")
            logger.warning(f"entering vairable is {entering}")
            self.basis[leaving] = entering
            logger.warning(f"basis is {self.basis}")

            #readjusting variables
            self.B_inv = np.linalg.inv(self.A_matrix[:, self.basis])
            self.current_value = self.objective_function.dot(x)


    def convert_numpy(self):
        self.A_matrix = np.array(self.A_matrix)
        self.basis = np.array(self.basis)
        self.B_inv = np.array(self.B_inv)
        self.bound = np.array(self.bound)
        self.rhs = np.array(self.rhs)
        self.objective_function = np.array(self.objective_function)

    def runner(self):
        self.read_constraints_from_file(file_path)
        self.adjust_for_slack_variables()
        self.adjust_bounds()
        self.convert_numpy()
        x_val = self.current_revised_simplex_elements()
        solution = self.objective_function.dot(x_val)
        return x_val, solution

    def _print_current_values(self, x):
        print("b inverse is \n", self.B_inv)
        print("current basis is ", self.basis)
        print("x values are ", x)
        print("Z = ", self.objective_function.dot(x))



lin_opt_obj = LinearOpt()
x, obj = lin_opt_obj.runner()


logger.error(obj)
logger.error(x)
