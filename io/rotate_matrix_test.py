# Last-modified: 22 Apr 2014 01:50:15
import numpy as np
import timeit
from math import cos, sin, sqrt
from scipy import weave

# http://stackoverflow.com/a/12261243/560844

def rotation_matrix_weave(axis, theta, mat = None):
    if mat == None:
        mat = np.eye(3,3)
    support = "#include <math.h>"
    code = """
        double x = sqrt(axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2]);
        double a = cos(theta / 2.0);
        double b = -(axis[0] / x) * sin(theta / 2.0);
        double c = -(axis[1] / x) * sin(theta / 2.0);
        double d = -(axis[2] / x) * sin(theta / 2.0);

        mat[0] = a*a + b*b - c*c - d*d;
        mat[1] = 2 * (b*c - a*d);
        mat[2] = 2 * (b*d + a*c);

        mat[3*1 + 0] = 2*(b*c+a*d);
        mat[3*1 + 1] = a*a+c*c-b*b-d*d;
        mat[3*1 + 2] = 2*(c*d-a*b);

        mat[3*2 + 0] = 2*(b*d-a*c);
        mat[3*2 + 1] = 2*(c*d+a*b);
        mat[3*2 + 2] = a*a+d*d-b*b-c*c;
    """
    weave.inline(code, ['axis', 'theta', 'mat'], support_code = support, libraries = ['m'])
    return(mat)


def rotation_matrix_numpy(axis, theta):
    mat = np.eye(3,3)
    axis = axis/sqrt(np.dot(axis, axis))
    a = cos(theta/2.)
    b, c, d = -axis*sin(theta/2.)
    return(np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                  [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                  [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]]))

def get_rotation_matrix(theta, rot_vector):
    """
    Given an angle theta and a 3D vector rot_vector, this routine
    computes the rotation matrix corresponding to rotating theta
    radians about rot_vector.

    Parameters
    ----------
    theta : scalar
        The angle in radians.

    rot_vector : array_like
        The axis of rotation.  Must be 3D.

    Returns
    -------
    rot_matrix : ndarray
         A new 3x3 2D array.  This is the representation of a
         rotation of theta radians about rot_vector in the simulation
         box coordinate frame

    See Also
    --------
    ortho_find

    Examples
    --------
    >>> a = [0,1,0]
    >>> theta = 0.785398163  # pi/4
    >>> rot = mu.get_rotation_matrix(theta,a)
    >>> rot
    array([[ 0.70710678,  0.        ,  0.70710678],
           [ 0.        ,  1.        ,  0.        ],
           [-0.70710678,  0.        ,  0.70710678]])
    >>> np.dot(rot,a)
    array([ 0.,  1.,  0.])
    # since a is an eigenvector by construction
    >>> np.dot(rot,[1,0,0])
    array([ 0.70710678,  0.        , -0.70710678])
    """

    ux = rot_vector[0]
    uy = rot_vector[1]
    uz = rot_vector[2]
    cost = np.cos(theta)
    sint = np.sin(theta)

    R = np.array([[cost+ux**2*(1-cost), ux*uy*(1-cost)-uz*sint, ux*uz*(1-cost)+uy*sint],
                  [uy*ux*(1-cost)+uz*sint, cost+uy**2*(1-cost), uy*uz*(1-cost)-ux*sint],
                  [uz*ux*(1-cost)-uy*sint, uz*uy*(1-cost)+ux*sint, cost+uz**2*(1-cost)]])

    return R


if __name__ == "__main__":
    v = np.array([3,5,0])
    axis = np.array([4,4,1])
    theta = 1.2 #radian
    m = rotation_matrix_numpy(axis, theta)
    print m
    m = rotation_matrix_weave(axis, theta)
    print m
