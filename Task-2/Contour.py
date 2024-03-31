import itertools
from typing import Tuple
import cv2
import numpy as np
from typing import Union
def iterate_contour(source: np.ndarray, contour_x: np.ndarray, contour_y: np.ndarray,
                    external_energy: np.ndarray, window_coordinates: list,
                    alpha: float, beta: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    :param source: image source
    :param contour_x: list of x coordinates of the contour
    :param contour_y: list of y coordinates of the contour
    :param alpha: factor multiplied to E_cont term in internal energy
    :param beta: factor multiplied to E_curv term in internal energy
    :param external_energy: Image Energy (E_line + E_edge)
    :param window_coordinates: array of window coordinates for each pixel
    :return:
    """

    src = np.copy(source)
    cont_x = np.copy(contour_x)
    cont_y = np.copy(contour_y)

    contour_points = len(cont_x)

    for Point in range(contour_points):
        MinEnergy = np.inf
        TotalEnergy = 0
        NewX = None
        NewY = None
        for Window in window_coordinates:
            # Create Temporary Contours With Point Shifted To A Coordinate
            CurrentX, CurrentY = np.copy(cont_x), np.copy(cont_y)
            CurrentX[Point] = CurrentX[Point] + Window[0] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
            CurrentY[Point] = CurrentY[Point] + Window[1] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

            # Calculate Energy At The New Point
            try:
                TotalEnergy = - external_energy[CurrentY[Point], CurrentX[Point]] + calculate_internal_energy(CurrentX,
                                                                                                              CurrentY,
                                                                                                              alpha,
                                                                                                              beta)
            except:
                pass

            # Save The Point If It Has The Lowest Energy In The Window
            if TotalEnergy < MinEnergy:
                MinEnergy = TotalEnergy
                NewX = CurrentX[Point] if CurrentX[Point] < src.shape[1] else src.shape[1] - 1
                NewY = CurrentY[Point] if CurrentY[Point] < src.shape[0] else src.shape[0] - 1

        # Shift The Point In The Contour To It's New Location With The Lowest Energy
        cont_x[Point] = NewX
        cont_y[Point] = NewY

    return cont_x, cont_y


def create_square_contour(source, num_xpoints, num_ypoints):
    """
    Create a square shape to be the initial contour
    :param source: image source
    :return: list of x points coordinates, list of y points coordinates and list of window coordinates
    """
    step = 5

    # Create x points lists
    t1_x = np.arange(0, num_xpoints, step)
    t2_x = np.repeat((num_xpoints) - step, num_xpoints // step)
    t3_x = np.flip(t1_x)
    t4_x = np.repeat(0, num_xpoints // step)

    # Create y points list
    t1_y = np.repeat(0, num_ypoints // step)
    t2_y = np.arange(0, num_ypoints, step)
    t3_y = np.repeat(num_ypoints - step, num_ypoints // step)
    t4_y = np.flip(t2_y)

    # Concatenate all the lists in one array
    contour_x = np.array([t1_x, t2_x, t3_x, t4_x]).ravel()
    contour_y = np.array([t1_y, t2_y, t3_y, t4_y]).ravel()

    # Shift the shape to a specific location in the image
    # contour_x = contour_x + (source.shape[1] // 2) - 85
    contour_x = contour_x + (source.shape[1] // 2) - 95
    contour_y = contour_y + (source.shape[0] // 2) - 55

    # Create neighborhood window
    WindowCoordinates = GenerateWindowCoordinates(5)

    return contour_x, contour_y, WindowCoordinates


def create_elipse_contour(source, num_points):
    """
        Represent the snake with a set of n points
        Vi = (Xi, Yi) , where i = 0, 1, ... n-1
    :param source: Image Source
    :param num_points: number of points to create the contour with
    :return: list of x coordinates, list of y coordinates and list of window coordinates
    """

    # Create x and y lists coordinates to initialize the contour
    t = np.arange(0, num_points / 10, 0.1)

    # Coordinates for CT Head.jpg image
    contour_x = (source.shape[1] // 2) + 180 * np.cos(t) 
    contour_y = (source.shape[0] // 2) + 190 * np.sin(t) 

    # Coordinates for fish.png image
    # contour_x = (source.shape[1] // 2) + 215 * np.cos(t)
    # contour_y = (source.shape[0] // 2) + 115 * np.sin(t) - 10

    contour_x = contour_x.astype(int)
    contour_y = contour_y.astype(int)

    # Create neighborhood window
    WindowCoordinates = GenerateWindowCoordinates(5)

    return contour_x, contour_y, WindowCoordinates


def GenerateWindowCoordinates(Size: int):
    """
    Generates A List of All Possible Coordinates Inside A Window of Size "Size"
    if size == 3 then the output is like this:
    WindowCoordinates = [[1, 1], [1, 0], [1, -1], [0, 1], [0, 0], [0, -1], [-1, 1], [-1, 0], [-1, 1], [2, 2]]

    :param Size: Size of The Window
    :return Coordinates: List of All Possible Coordinates
    """

    # Generate List of All Possible Point Values Based on Size
    Points = list(range(-Size // 2 + 1, Size // 2 + 1))
    PointsList = [Points, Points]

    # Generates All Possible Coordinates Inside The Window
    Coordinates = list(itertools.product(*PointsList))
    return Coordinates


def calculate_internal_energy(CurrentX, CurrentY, alpha: float, beta: float):
    """
    The internal energy is responsible for:
        1. Forcing the contour to be continuous (E_cont)
        2. Forcing the contour to be smooth     (E_curv)
        3. Deciding if the snake wants to shrink/expand

    Internal Energy Equation:
        E_internal = E_cont + E_curv

    E_cont
        alpha * ||dc/ds||^2

        - Minimizing the first derivative.
        - The contour is approximated by N points P1, P2, ..., Pn.
        - The first derivative is approximated by a finite difference:

        E_cont = | (Vi+1 - Vi) | ^ 2
        E_cont = (Xi+1 - Xi)^2 + (Yi+1 - Yi)^2

    E_curv
        beta * ||d^2c / d^2s||^2

        - Minimizing the second derivative
        - We want to penalize if the curvature is too high
        - The curvature can be approximated by the following finite difference:

        E_curv = (Xi-1 - 2Xi + Xi+1)^2 + (Yi-1 - 2Yi + Yi+1)^2

    ==============================

    Alpha and Beta
        - Small alpha make the energy function insensitive to the amount of stretch
        - Big alpha increases the internal energy of the snake as it stretches more and more

        - Small beta causes snake to allow large curvature so that snake will curve into bends in the contour
        - Big beta leads to high price for curvature so snake prefers to be smooth and not curving

    :return:
    """
    JoinedXY = np.array((CurrentX, CurrentY))
    Points = JoinedXY.T

    # Continuous  Energy
    PrevPoints = np.roll(Points, 1, axis=0)
    NextPoints = np.roll(Points, -1, axis=0)
    Displacements = Points - PrevPoints
    PointDistances = np.sqrt(Displacements[:, 0] ** 2 + Displacements[:, 1] ** 2)
    MeanDistance = np.mean(PointDistances)
    ContinuousEnergy = np.sum((PointDistances - MeanDistance) ** 2)

    # Curvature Energy
    CurvatureSeparated = PrevPoints - 2 * Points + NextPoints
    Curvature = (CurvatureSeparated[:, 0] ** 2 + CurvatureSeparated[:, 1] ** 2)
    CurvatureEnergy = np.sum(Curvature)

    return alpha * ContinuousEnergy + beta * CurvatureEnergy


def calculate_external_energy(source, WLine, WEdge):
    """
    The External Energy is responsible for:
        1. Attracts the contour towards the closest image edge with dependence on the energy map.
        2. Determines whether the snake feels attracted to object boundaries

    An energy map is a function f (x, y) that we extract from the image – I(x, y):

        By given an image – I(x, y), we can build an energy map – f(x, y),
        that will attract our snake to edges on our image.

    External Energy Equation:
        E_external = w_line * E_line + w_edge * E_edge

    E_line
        I(x, y)
        Smoothing filter could be applied to I(x, y) to remove noise
        Depending on the sign of w_line the snake will be attracted either to bright lines or dark lines

    E_curv
        -|| Gradiant(I(x,y)) ||^2
    ==============================
    :param source: Image source
    :param WLine: weight of E_line term
    :param WEdge: weight of E_edge term
    :return:
    """
    src = np.copy(source)
    # convert to gray scale if not already
    if len(src.shape) > 2:
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    else:
        gray = src

    # Apply Gaussian Filter to smooth the image
    ELine = gaussian_filter(gray, 7, 7 * 7)
    # Get Gradient Magnitude & Direction
    EEdge, gradient_direction = sobel_edge(ELine, GetDirection=True)
    # print("Shape of ELine:", ELine.shape)
    # print("Shape of EEdge:", EEdge.shape)
    # Pad EEdge with zeros to match the shape of ELine
    pad_width = ((0, 0), (0, 0))  # Pad one row/column on each side
    EEdge_padded = np.pad(EEdge, pad_width, mode='constant', constant_values=0)
    # print("Shape of EEdge_padded:", EEdge_padded.shape)
    # EEdge *= 255 / EEdge.max()
    # EEdge = EEdge.astype("int16")

    return WLine * ELine + WEdge * EEdge_padded 


def sobel_edge(source: np.ndarray, GetMagnitude: bool = True, GetDirection: bool = False):
    """
        Apply Sobel Operator to detect edges
        :param source: Image to detect edges in
        :param GetMagnitude: Get Magnitude of horizontal and vertical edges
        :param GetDirection: Get Gradient direction in Pi Terms
        :return: edges image
    """
    if len(source.shape) > 2:
        gray = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY)
    else:
        gray = source

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Compute gradient magnitude
    mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)


    if GetMagnitude:
        if GetDirection:
            # Compute gradient direction
            direction = np.arctan2(sobel_y, sobel_x)
            return mag, direction
        else:
            return mag
    elif GetDirection:
        direction = np.arctan2(sobel_y, sobel_x)
        return direction
    else:
        return None
def gaussian_filter(source: np.ndarray, shape: int = 5, sigma: Union[int, float] = 64) -> np.ndarray:
    """
        Gaussian Low Pass Filter Implementation
    :param source: Image to Apply Filter to
    :param shape: An Integer that denotes th Kernel size if 3
                  then the kernel is (3, 3)
    :param sigma: Standard Deviation
    :return: Filtered Image
    """
    src = np.copy(source)
    blurred = cv2.GaussianBlur(src, (shape, shape), sigma)
    return blurred

#     # Parameters For hand_256.png image
#     alpha = 20  # Continuous
#     beta = 0.3  # Curvature
#     gamma = 5  # External
#     w_line = 1  # E_line
#     w_edge = 8  # E_edge
#     num_points_circle = 65
#     num_xpoints = 180
#     num_ypoints = 180
#     num_iterations = 105


# parameters for CT Head.png image
#     alpha = 0.2  # Continuous
#     beta = 1.1 # Curvature
#     gamma = 4.5  # External
#     w_line = 1  # E_line
#     w_edge = 8  # E_edge
#     num_points_circle = 60
#     num_xpoints = 180
#     num_ypoints = 180
#     num_iterations = 90