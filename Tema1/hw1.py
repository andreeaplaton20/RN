import pathlib
import re
import math
from copy import deepcopy


def load_system(path: pathlib.Path) -> tuple[list[list[float]], list[float]]:
    text = path.read_text(encoding='utf-8')
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    variables = sorted(set(re.findall(r'[a-z]', text)))

    A, B = [], []

    for line in lines:
        left, right = line.split('=')
        left, right = left.strip(), right.strip()

        coefficients = []

        for variable in variables:
            match = re.search(rf'([+-]?\s*\d*\.?\d*)\s*{variable}', left)

            if match:
                val = match.group(1).replace(' ', '')
                if val in ('+', ''):
                    val = 1
                elif val == '-':
                    val = -1
                else:
                    val = int(val)
            else:
                val = 0

            coefficients.append(int(val))

        A.append(coefficients)
        B.append(int(right))

    return A, B

A, B = load_system(pathlib.Path("system.txt"))
print(f"{A=} {B=}")


# det(A)=a11(a22a33−a23a32)−a12(a21a33−a23a31)+a13(a21a32−a22a31)
def determinant(matrix: list[list[float]]) -> float:
    a, b, c = matrix[0]
    d, e, f = matrix[1]
    g, h, i = matrix[2]
    return a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g)
print(f"{determinant(A)=}")


# Trace(A)=a11+a22+a33
def trace(matrix: list[list[float]]) -> float:
    tr = 0
    for elem in range(len(matrix)):
        tr += matrix[elem][elem]
    return tr

print(f"{trace(A)=}")


# ||B||=b21+b22+b23−−−−−−−−−√
def norm(vector: list[float]) -> float:
    n = 0
    for i in vector:
        n += pow(i, 2)
    return math.sqrt(n)

print(f"{norm(B)=}")


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    trans = []
    for row in matrix:
        trans.append(row)
    temp = zip(*trans)
    transposed = []
    for elem in temp:
        transposed.append(list(elem))

    return transposed

print(f"{transpose(A)=}")


'''
How to perform matrix-vector multiplication     
    Check compatibility: First, ensure the number of columns in the matrix m x n 
is equal to the number of rows in the vector (a column vector with n entries). 
    Calculate the resulting vector: The resulting vector will have a 
number of rows equal to the number of rows in the original matrix m. 
    Step 1: The first entry. Multiply the first row of the matrix by the vector. 
Add the products of each corresponding element. 
    Step 2: Subsequent entries. Repeat this dot product for each 
subsequent row of the matrix to find each new entry in the resulting vector.
'''
def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if len(matrix) != len(vector):
        raise ValueError('Matrix must have same number of columns')
    result = []
    suma = 0
    for row in matrix:
        for i in range(len(vector)):
            suma += row[i] * vector[i]
        result.append(suma)
        suma = 0

    return result

print(f"{multiply(A, B)=}")



def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:
    if determinant(matrix) == 0:
        raise ValueError('Matrix must not have the determinant equal to 0')

    result = []

    for i in range(len(matrix)):
        new_matrix = deepcopy(matrix)
        for row in range(len(matrix)):
            new_matrix[row][i] = vector[row]
        result.append(determinant(new_matrix)/determinant(matrix))


    return result

print(f"{solve_cramer(A, B)=}")



def determinant2(matrix: list[list[float]]) -> float:
    a, b = matrix[0]
    c, d = matrix[1]
    return a*d - b*c

def minor(matrix: list[list[float]], i: int, j: int) -> list[list[float]]:
    result = []
    for row in range(len(matrix)):
        if row == i:
            continue
        new_row = []
        for column in range(len(matrix)):
            if column == j:
                continue
            new_row.append(matrix[row][column])
        result.append(new_row)
    return result

def cofactor(matrix: list[list[float]]) -> list[list[float]]:
    result = deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            result[i][j] = pow(-1, i+j) * determinant2(minor(matrix, i, j))
    return result

def adjoint(matrix: list[list[float]]) -> list[list[float]]:
    return transpose(cofactor(matrix))

def solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    adj = adjoint(matrix)
    new_matrix = deepcopy(matrix)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            new_matrix[i][j] = adj[i][j] / determinant(matrix)
    return multiply(new_matrix, vector)

print(f"{solve(A, B)=}")