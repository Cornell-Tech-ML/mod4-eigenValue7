import math
import random
from dataclasses import dataclass
from typing import List, Tuple


def make_pts(N: int) -> List[Tuple[float, float]]:
    """Generates a list of N random points in 2D space.

    This function generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the random.random() function, which returns a random floating point number in the range [0.0, 1.0).

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        List[Tuple[float, float]]: A list of N tuples, each representing a point in 2D space.

    """
    X = []
    for i in range(N):
        x_1 = random.random()
        x_2 = random.random()
        X.append((x_1, x_2))
    return X


@dataclass
class Graph:
    N: int
    X: List[Tuple[float, float]]
    y: List[int]


def simple(N: int) -> Graph:
    """Generates a simple dataset for binary classification.

    This function generates a simple dataset for binary classification. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the make_pts function. The labels for the points are generated based on the x-coordinate of the point. If the x-coordinate is less than 0.5, the label is 1, otherwise it is 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def diag(N: int) -> Graph:
    """Generates a dataset for binary classification based on a diagonal decision boundary.

    This function generates a dataset for binary classification where the decision boundary is a diagonal line in the 2D space. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the make_pts function. The labels for the points are generated based on their position relative to the diagonal line. If the point is above the diagonal line, the label is 1, otherwise it is 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 + x_2 < 0.5 else 0
        y.append(y1)
    return Graph(N, X, y)


def split(N: int) -> Graph:
    """Generates a dataset for binary classification with a split decision boundary.

    This function generates a dataset for binary classification where the decision boundary is a split line in the 2D space. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the make_pts function. The labels for the points are generated based on their position relative to the split line. If the point is to the left or right of the split line, the label is 1, otherwise it is 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if x_1 < 0.2 or x_1 > 0.8 else 0
        y.append(y1)
    return Graph(N, X, y)


def xor(N: int) -> Graph:
    """Generates a dataset for binary classification with an XOR decision boundary.

    This function generates a dataset for binary classification where the decision boundary is an XOR operation in the 2D space. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the make_pts function. The labels for the points are generated based on their position relative to the XOR operation. If the point is in the top-left or bottom-right quadrant, the label is 1, otherwise it is 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        y1 = 1 if ((x_1 < 0.5 and x_2 > 0.5) or (x_1 > 0.5 and x_2 < 0.5)) else 0
        y.append(y1)
    return Graph(N, X, y)


def circle(N: int) -> Graph:
    """Generates a dataset for binary classification with a circular decision boundary.

    This function generates a dataset for binary classification where the decision boundary is a circle in the 2D space. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using the make_pts function. The labels for the points are generated based on their position relative to the circle. If the point is inside the circle, the label is 1, otherwise it is 0.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """
    X = make_pts(N)
    y = []
    for x_1, x_2 in X:
        x1, x2 = (x_1 - 0.5, x_2 - 0.5)
        y1 = 1 if x1 * x1 + x2 * x2 > 0.1 else 0
        y.append(y1)
    return Graph(N, X, y)


def spiral(N: int) -> Graph:
    """Generates a dataset for binary classification with a spiral decision boundary.

    This function generates a dataset for binary classification where the decision boundary is a spiral in the 2D space. It generates N random points in the 2D space, where each point is represented by a tuple of two floats between 0 and 1. The points are generated using a spiral pattern, with the first half of the points forming one spiral and the second half forming another spiral that intersects the first one. The labels for the points are generated based on their position relative to the spiral. If the point is on the first spiral, the label is 0, otherwise it is 1.

    Args:
    ----
        N (int): The number of points to generate.

    Returns:
    -------
        Graph: A Graph object containing the generated points and their labels.

    """

    def x(t: float) -> float:
        return t * math.cos(t) / 20.0

    def y(t: float) -> float:
        return t * math.sin(t) / 20.0

    X = [
        (x(10.0 * (float(i) / (N // 2))) + 0.5, y(10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    X = X + [
        (y(-10.0 * (float(i) / (N // 2))) + 0.5, x(-10.0 * (float(i) / (N // 2))) + 0.5)
        for i in range(5 + 0, 5 + N // 2)
    ]
    y2 = [0] * (N // 2) + [1] * (N // 2)
    return Graph(N, X, y2)


datasets = {
    "Simple": simple,
    "Diag": diag,
    "Split": split,
    "Xor": xor,
    "Circle": circle,
    "Spiral": spiral,
}
