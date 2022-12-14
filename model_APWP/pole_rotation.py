import cartopy.crs as ccrs
from scipy.constants import Julian_year
import numpy as np

d2r = np.pi/180
r2d = 180/np.pi

def spherical_to_cartesian(longitude, latitude, norm = 1):
    colatitude = 90. - latitude
    return np.array([norm * np.sin(colatitude * d2r) * np.cos(longitude * d2r),
                     norm * np.sin(colatitude * d2r) * np.sin(longitude * d2r),
                     norm * np.cos(colatitude * d2r)])

def cartesian_to_spherical(vecs):
    v = np.reshape(vecs, (3, -1))
    norm = np.sqrt(v[0, :] * v[0, :] + v[1, :] * v[1, :] + v[2, :] * v[2, :])
    latitude = 90. - np.arccos(v[2, :] / norm) * r2d
    longitude = np.arctan2(v[1, :],v[0, :]) * r2d
    
    return longitude, latitude, norm

def construct_euler_rotation_matrix(alpha, beta, gamma):
    """
    Make a 3x3 matrix which represents a rigid body rotation,
    with alpha being the first rotation about the z axis,
    beta being the second rotation about the y axis, and
    gamma being the third rotation about the z axis.

    All angles are assumed to be in radians
    """
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha), 0.],
                          [np.sin(alpha), np.cos(alpha), 0.],
                          [0., 0., 1.]])
    rot_beta = np.array([[np.cos(beta), 0., np.sin(beta)],
                         [0., 1., 0.],
                         [-np.sin(beta), 0., np.cos(beta)]])
    rot_gamma = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                          [np.sin(gamma), np.cos(gamma), 0.],
                          [0., 0., 1.]])
    rot = np.dot(rot_gamma, np.dot(rot_beta, rot_alpha))
    return rot

class Pole(object):
    """
    Class representing a pole on the globe:
    essentially a 3-vector with some additional
    properties and operations.
    """

    def __init__(self, longitude, latitude, magnitude=1.0, A95=None):
        """
        Initialize the pole with lon, lat, and A95 uncertainty. Removed norm from Rose version, here we assume everything is unit vector. 
        longitude, latitude, and A95 are all taken in degrees.
        """

        self._pole = np.ndarray.flatten(spherical_to_cartesian(longitude, latitude, magnitude)) # pole position in cartesian coordinates, easier for addition operations
        self._A95 = A95

    @property
    def longitude(self):
        return np.arctan2(self._pole[1], self._pole[0]) * r2d

    @property
    def latitude(self):
        return 90. - np.arccos(self._pole[2] / self.magnitude) * r2d

    @property
    def colatitude(self):
        return np.arccos(self._pole[2] / self.magnitude) * r2d

    @property
    def magnitude(self):
        return np.sqrt(self._pole[0] * self._pole[0] + self._pole[1] * self._pole[1] + self._pole[2] * self._pole[2])

#     @property
#     def angular_error(self):
#         return self._angular_error

    def copy(self):
        return copy.deepcopy(self)

    def rotate(self, pole, angle):
        # The idea is to rotate the pole about a given pole
        # at the pole of the coordinate system, then perform the
        # requested rotation, then restore things to the original
        # orientation
        
        
        p = pole._pole
        
        lon, lat, mag = cartesian_to_spherical(p)
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(
            -lon[0] * d2r, -colat[0] * d2r, angle * d2r)
        
        m2 = construct_euler_rotation_matrix(
            0., colat[0] * d2r, lon[0] * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
    

    def _rotate(self, pole, angle):
        print(self.longitude, self.latitude)
        p = pole._pole
        
        lon, lat, _ = cartesian_to_spherical(p)
        lon = T.as_tensor_variable(lon[0])
        lat = T.as_tensor_variable(lat[0])
        
        colat = 90. - lat
        m1 = construct_euler_rotation_matrix(-lon * d2r, -colat * d2r, angle * d2r)
        m2 = construct_euler_rotation_matrix(0., colat * d2r, lon * d2r)
        self._pole = np.dot(m2, np.dot(m1, self._pole))
        self.longitude = cartesian_to_spherical(self._pole.tolist())[0].tolist()[0]
        self.latitude = cartesian_to_spherical(self._pole.tolist())[1].tolist()[0]
        self._pole = spherical_to_cartesian(self.longitude, self.latitude, self.magnitude)

        self.colatitude = 90 - self.latitude
        
    def add(self, pole):
        self._pole = self._pole + pole._pole

    def plot(self, axes, south_pole=False, **kwargs):
        artists = []
        if self._A95 is not None:
            lons = np.linspace(0, 360, 360)
            lats = np.ones_like(lons) * (90. - self._A95)
            magnitudes = np.ones_like(lons)
            
            vecs = spherical_to_cartesian(lons, lats, magnitudes)
            rotation_matrix = construct_euler_rotation_matrix(
                0., (self.colatitude) * d2r, self.longitude * d2r)
            rotated_vecs = np.dot(rotation_matrix, vecs)
            lons, lats, magnitudes = cartesian_to_spherical(rotated_vecs.tolist())
            if south_pole is True:
                lons = lons-180.
                lats = -lats
            path = matplotlib.path.Path(np.array([lons, lats]).T)
            circ_patch = matplotlib.patches.PathPatch(
                path, transform=ccrs.Geodetic(), alpha=0.5, **kwargs)
            circ_artist = axes.add_patch(circ_patch)
            artists.append(circ_artist)
        if south_pole is False:
            artist = axes.scatter(self.longitude, self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        else:
            artist = axes.scatter(self.longitude-180., -self.latitude,
                                  transform=ccrs.PlateCarree(), **kwargs)
        artists.append(artist)
        return artists

class PaleomagneticPole(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, age=0., sigma_age=0.0, **kwargs):

        if np.iterable(sigma_age) == 1:
            assert len(sigma_age) == 2  # upper and lower bounds
            self._age_type = 'uniform'
        else:
            self._age_type = 'gaussian'

        self._age = age
        self._sigma_age = sigma_age

        super(PaleomagneticPole, self).__init__(
            longitude, latitude, 1.0, **kwargs)

class EulerPole(Pole):
    """
    Subclass of Pole which represents an Euler pole.
    The rate is given in deg/Myr
    
    Here we send the rotation rate in radian/sec to the father class as the magnitude. 
    """

    def __init__(self, longitude, latitude, rate, **kwargs):
        r = rate * d2r / Julian_year / 1.e6
        super(EulerPole, self).__init__(longitude, latitude, magnitude = r, **kwargs)

    @property
    def rate(self):
        # returns the angular velocity of the object that is rotating about a given Euler pole
        return self.magnitude * r2d * Julian_year * 1.e6

    def angle(self, time):
        return self.rate * time

    def speed_at_point(self, pole):
        """
        Given a point, calculate the speed that point
        rotates around the Euler pole. This assumes that
        the test pole has a radius equal to the radius of Earth,
        6371.e3 meters. It returns the speed in cm/yr.
        """
        # Give the point the radius of the earth
        point = pole._pole
        point = point / np.sqrt(np.dot(point, point)) * 6371.e3
#         print(np.array([point[0], point[1], point[2]]))
        # calculate the speed
        vel = np.cross(self._pole, np.array([point[0], point[1], point[2]]))    
        speed = np.sqrt(np.dot(vel, vel))

        return speed * Julian_year * 100.


class PlateCentroid(Pole):
    """
    Subclass of Pole which represents the centroid
    of a plate. Proxy for plate position (since the
    plate is itself an extended object).
    """

    def __init__(self, longitude, latitude, **kwargs):
        super(PlateCentroid, self).__init__(
            longitude, latitude, 6371.e3, **kwargs)