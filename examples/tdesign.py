import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Enables 3D plotting
from scipy.special import lpmv
import math
from fcmaes.optimizer import wrapper, Bite_cpp
from scipy.optimize import Bounds
from fcmaes import retry, de
from loguru import logger
import sys
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {message}", level="INFO")

np.set_printoptions(legacy='1.25') # required for newest numpy on Python 12, comment out if it doesn't work

# Convert Cartesian coordinates to spherical coordinates
def cartesian_to_spherical(points):
    spherical_coords = np.empty((len(points),2))
    for i, (x, y, z) in enumerate(points):
        theta = np.arccos(z)  # Polar angle
        phi = np.arctan2(y, x)  # Azimuthal angle
        if phi < 0: 
            phi += 2*np.pi
        spherical_coords[i] = (theta, phi)
    return spherical_coords

def spherical_to_cartesian(theta, phi):
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    return x, y, z

def x_to_points(x): # stack theta, phi into an array of shape (N,2)
    N = len(x)//2
    return np.stack([x[:N], x[N:]], axis=1)

def normalize_weights_to_average_one(weights):
    weights = np.array(weights)
    N = len(weights)
    sum = np.sum(weights)
    if sum == 0:
        return np.ones(N)
    alpha = N / sum
    return alpha * weights

def fibonacci_sphere(N):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # Golden angle
    for i in range(N):
        y = 1 - (i / float(N - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)          # radius at y
        theta = phi * i                      # golden angle increment
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

def real_spherical_harmonics(l_max, theta, phi):
    """
    Compute fully normalized real spherical harmonics up to degree l_max
    using scipy.special.lpmv.

    The real spherical harmonics are defined by:
    
      For m = 0:
         Y_{l,0}(\theta,phi) = N_{l0} * P_l^0(cos(theta))
      For m > 0:
         Y_{l,m}(\theta,phi) = sqrt(2)*N_{lm} * P_l^m(cos(theta)) * cos(m*phi)
         Y_{l,-m}(\theta,phi)= sqrt(2)*N_{lm} * P_l^m(cos(theta)) * sin(m*phi)
         
    where the normalization factor is:
    
         N_{lm} = (-1)^m * sqrt((2*l+1)/(4*pi) * (l-m)!/(l+m)!)
    
    Parameters
    ----------
    l_max : int
        Maximum degree of spherical harmonics.
    theta : array_like
        Polar angles (in radians), shape (N,).
    phi : array_like
        Azimuthal angles (in radians), shape (N,).

    Returns
    -------
    Y : dict
        A dictionary with keys (l, m) where l = 0,...,l_max and m = -l,...,l.
        Each entry is a NumPy array of shape (N,) containing the value of the
        corresponding spherical harmonic.
    """
    # Ensure theta and phi are numpy arrays.
    theta = np.asarray(theta)
    phi   = np.asarray(phi)
    # Compute cos(theta) for the Legendre functions.
    cos_theta = np.cos(theta)

    Y = {}  # dictionary to store the spherical harmonics
    for l in range(l_max+1):
        for m in range(l+1):
            # Compute the normalization factor for fully normalized spherical harmonics.
            norm = np.sqrt((2*l+1)/(4*np.pi) * math.factorial(l-m)/math.factorial(l+m))
            # Compute the associated Legendre function for order m and degree l.
            P_lm = lpmv(m, l, cos_theta)           
            if m == 0:
                # For m = 0, no extra trigonometric factor is needed.
                Y[(l, 0)] = norm * P_lm
            else:
                # For m > 0, compute both the cosine and sine components.
                Y[(l, m)]  = np.sqrt(2) * norm * P_lm * np.cos(m * phi)
                Y[(l, -m)] = np.sqrt(2) * norm * P_lm * np.sin(m * phi)
    return Y

def weighted_spherical_harmonics(l_max, theta, phi, weights):
    """
    Given:
      - l_max: maximum spherical-harmonic degree,
      - theta, phi: arrays of length N with the spherical angles of your points,
      - weights: array of length N with the corresponding weights w_i,
    
    compute the 'integral' (weighted sum) of each spherical harmonic:
        I[l,m] = sum_i( w_i * Y_{l,m}(theta_i, phi_i) ).
    
    Returns a dict integrals[(l,m)] = floating-point result.
    """    
    weights = normalize_weights_to_average_one(weights)
    # 1) Compute the SH values at all points (unweighted).
    Y = real_spherical_harmonics(l_max, theta, phi)
    # 2) Multiply each Y_{l,m} by the corresponding weight and sum up.
    W = {}
    for l in range(l_max+1):
        for m in range(-l, l+1):
            # element-wise multiply by w_i and sum
            W[(l,m)] = np.sum(Y[(l,m)] * weights)
    return W

def symmetry_error(Y, N, l_max):
    """
    For each degree l from 0 to l_max, compute the sum over m of the square of the
    (pointwise) sums of the spherical harmonic values. Then multiply by 4*pi/(N^2).
    
    Parameters:
      Y : dict mapping (l, m) -> array of shape (N,)
      N : int, number of points
      l_max : maximum degree
      
    Returns:
      s : numpy array of shape (l_max+1,)
    """
    s = np.zeros(l_max+1)
    # For l = 0 (only m=0 exists)
    s[0] = np.abs(np.sum(Y[(0, 0)]))**2
    for l in range(1, l_max+1):
        for m in range(-l, l+1):
            s[l] += np.abs(np.sum(Y[(l, m)]))**2
    s[np.abs(s) < 1.e-20] = 0.
    return s * 4*np.pi / (N**2)

def symmetry(pts, l_max, weights=None):
    """
    Compute a symmetry measure for a set of points.
    pts has 2 columns, it is assumed to be [theta, phi];
    
    Returns:
      An array of length l_max+1.
    """
    pts = np.array(pts)
    # Assume pts[:,0]=theta, pts[:,1]=phi
    if weights is None:
        Y = real_spherical_harmonics(l_max, pts[:, 0], pts[:, 1])
    else:
        Y = weighted_spherical_harmonics(l_max, pts[:, 0], pts[:, 1], weights)
    N = pts.shape[0]
    error = symmetry_error(Y, len(pts), l_max)
    # Create a multiplier: 1 / (2*l + 1) for l=0,...,l_max
    mult = 1. / (2*np.arange(0, l_max+1) + 1)
    return error * mult

def t_design_error(points, l_max, weights=None):
    syms = symmetry(points, l_max, weights)
    return sum(syms[1:l_max+1])

def visualize_points(points, l_max, weights=None):
    """
    Visualize points on the unit sphere.
    
    Each point in 'points' should be a tuple:
      - (theta, phi) or
      - (theta, phi, weight)
    where theta is in [0, pi] and phi is in [0, 2*pi]. If weight is not provided,
    all points are assumed to have equal weight.
    
    Parameters:
    -----------
    points : list or array-like
        A list of points in spherical coordinates.
    """
    print(f'{len(points)} points (theta,phi):')
    for p in points:
        print(list(p)) 
    print()
    syms = symmetry(points, l_max+3, weights)
    weights = normalize_weights_to_average_one(weights)
    print(f"weights: {list(weights)}")
    print("symetries = ", syms)
    print(f"symetry error = {sum(syms[1:l_max+1])}")


    # Unpack theta and phi
    theta, phi = zip(*points)
    weights = np.ones(len(points)) if weights is None else weights
    theta = np.array(theta)
    phi = np.array(phi)    
    # Convert spherical coordinates to Cartesian coordinates on the unit sphere.
    x, y, z = spherical_to_cartesian(theta, phi)  
    # Create a new figure with 3D axes.
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the points, using weights to color the points.
    scatter = ax.scatter(x, y, z, c=weights, cmap='viridis', s=50)
    
    # Optionally, plot a wireframe of the unit sphere for reference.
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 25)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color='gray', alpha=0.3)
    
    # Label the axes and add a title.
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Points on the Unit Sphere")
    
    # Add a colorbar to indicate the weight (if applicable).
    plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=10, label='Weight')   
    plt.show()
    
def optimize_weighted(N, l_max, workers=20, max_evals=150000, max_iters=1):
 
    def fit(x):
        points = x_to_points(x[:2*N])
        weights = x[2*N:]
        return t_design_error(points, l_max, weights)
    
    x0 = np.array(list(cartesian_to_spherical(fibonacci_sphere(N)).flatten()) + [1]*N) 
  
    dim = N*3 # # we encode the input by concatenating the theta, phi and weight vectors
    # apply BiteOpt using parallel restart 
    for i in range(max_iters):
        result = retry.minimize(wrapper(fit), 
                                bounds=Bounds([0]*dim,[np.pi]*N + [2*np.pi]*N + [2]*N), 
                                num_retries=workers, workers=workers,
                                stop_fitness = 0, 
                                optimizer=Bite_cpp(max_evals, guess=x0, stop_fitness=0))
    x0 = result.x
    points = x_to_points(result.x[:2*N])
    weights = result.x[2*N:]  
    print(f"Fitness: {fit(x0)}")        
    visualize_points(points, l_max, weights)
    
def show_results(N, l_max):

    # n=42 l_max = 10
    if N == 42:
        x = [2.447217284242543, 1.4517070680861281, 0.8907621209089062, 1.745862929099355, 1.4049928360005761, 2.123718951551832, 1.4353585379167855, 1.795193579750673, 0.9936439341556538, 2.42688314878376, 2.061148509982718, 0.6356259587455823, 0.9022157881722463, 0.4161660936306942, 2.0783535824454447, 1.5987738995397742, 1.914019919169261, 1.377057139594396, 1.8001113473988923, 1.9030536604204784, 2.2986410362170338, 1.1643720284459766, 1.0585497682737501, 2.049939403516931, 1.4679930768047569, 1.3719018453624228, 1.5103957757421282, 0.8567644078993933, 0.6087172979572448, 1.9802333048552152, 2.53426390482221, 2.5555780724806496, 1.4814488883229116, 0.38392536310433395, 0.912090306934545, 2.746761190754236, 2.5343394636596, 0.8750179958146024, 0.27975686369694425, 2.003199964794502, 1.1852333762660772, 3.0416516361988566, 1.0303914048801714, 5.604941266160562, 2.8593015359409786, 4.165114459914451, 3.238975312951073, 4.598896945880547, 1.5141547998093383, 5.096594585219709, 4.308797093290831, 5.258320077680217, 2.370264024699624, 4.915642480324595, 0.2816873934360353, 2.2568415894699476, 0.4743672484760749, 2.0519720817232328, 6.205336584239582, 6.162612488037494, 1.03323260646458, 3.5546001474686792, 3.925717844888819, 5.098644649280042, 2.1949792038737597, 1.6296533929828394, 4.610023292590779, 3.8328097495962488, 0.4550427435469594, 5.769891533320035, 1.024244687866028, 2.977183013149219, 3.1129991266539556, 2.0704647095249578, 2.6508963308118045, 3.8262559858367093, 1.6189186869012622, 4.417890964353101, 6.209840414334106, 3.571479923677028, 6.155149089475483, 5.6761304055135895, 0.9465296848185534, 0.986147115054497, 1.0489187668502167, 0.9883058617724704, 1.0175104066742564, 0.8824282372424973, 1.028793008294155, 0.9703961792070446, 1.002131602520189, 1.0270075743510283, 1.0290659808831222, 0.9615600697770987, 0.9879198676650326, 0.9515430765357046, 1.0707269724502493, 1.0502123486759751, 0.9266650076605122, 0.9761179116691323, 0.9078341584862649, 0.9840549407175323, 1.080419741801789, 0.9446067526126258, 0.8789817289291073, 1.00482022102977, 0.9526431862190545, 1.071467063467698, 0.9551061236646734, 0.9201279307842383, 1.03735587585606, 1.0761785477940142, 0.8943550214931897, 0.9622124579346069, 1.0544568341596356, 0.9448660275161783, 1.0498802276077306, 0.959678770416923, 0.8541431784018771, 0.874581824758353, 1.0690334043389604, 0.902560793239468, 1.0463302000088717, 0.9075955826429161, 1.0417308263615208, 0.9285647737770256]
    # n = 48 l_max = 11
    elif N == 48:
        x = [1.6609636528376754, 2.0332922687473345, 1.1699657873384723, 1.5932453811460041, 1.3536585641517274, 1.5483472724465486, 1.073587874358636, 1.1152535803821837, 2.0680047792324183, 2.010603450228788, 2.2297858837891393, 1.5345315259972416, 0.6118940481704037, 1.4806290007519498, 1.2769583766317139, 2.026339073211587, 0.6465941180951524, 0.8820545263724141, 1.1083003848440802, 1.8646342769603, 2.417935245016386, 0.9823648155982182, 1.1309892033599005, 1.4357013723570007, 2.2595381272207162, 2.343042644298124, 1.7106338266279957, 0.4739391602093199, 0.798550009287575, 2.5296986054189228, 2.6676534933792895, 1.9716268662511167, 0.7236574085731134, 1.430958826962819, 1.6070611275927884, 2.9552696856582408, 1.95310840349356, 0.6158321564280353, 0.18632296793217853, 1.7879340894413376, 2.159227837992271, 2.5257604971608534, 1.7058912812390827, 0.3396445414629753, 0.9118067698009803, 1.1884842500964654, 2.801948112129519, 2.4949985354932576, 0.9119797077545138, 5.747521258986248, 1.9593394388479823, 2.944141374358616, 1.3772699720045773, 6.085734027948687, 5.811813756189445, 3.218384860617152, 2.6702211025976563, 3.94640248438286, 0.7328782394911566, 5.498357122596625, 0.5836933335599276, 4.053572361345344, 0.2642265358820038, 0.07679220702698455, 2.211196627377644, 0.02809835738564689, 2.6059286053969393, 3.405819189473536, 1.5207000444048149, 5.229363129681135, 0.8048098307941682, 3.5814665921405044, 3.1696910109730685, 4.54147531114239, 1.8087779443138645, 5.670215480608386, 1.3998826575547507, 3.7252859871504094, 2.5286228270283955, 5.100932092434556, 4.662292697993756, 4.950370597903347, 2.356764469009016, 4.611592591253544, 1.335629954569688, 3.1071505132321624, 1.4699999376785504, 4.518862625591488, 2.0877704760905256, 6.2487431668184845, 0.43987393854952933, 4.082374620208072, 3.874470893082349, 4.477222608157907, 0.9407819666267304, 5.352789280961853, 0.9963376524542624, 1.120264022145831, 1.144560325811255, 1.1260309098059014, 1.125722042879106, 1.126030909783953, 0.9902878394999327, 0.9940290303494079, 0.9902878395067557, 1.1178972257806292, 1.186925139796034, 1.1238732280565524, 1.001467504946035, 0.9963376524882124, 0.9369118612307583, 0.9940290303826887, 1.0292063788379486, 0.8367037285617723, 1.1202640221549567, 0.9369118612583268, 0.953746083634209, 1.0298148736764168, 1.1178972257379727, 0.9165404722422674, 0.8367037285603006, 1.1225671138848146, 1.046127083566, 1.1761540769264995, 1.1225671138526414, 1.001467504956023, 1.1761540769375554, 1.144560325814979, 0.9537460836371334, 1.0461270835586611, 1.1238732280689725, 1.059451796083912, 1.0197256171954978, 1.0424226498126266, 1.0594517960687824, 1.1257220429168302, 1.029814873691106, 1.0424226497957263, 0.9165404722395816, 0.9661694216200659, 1.1869251397886573, 1.0197256172172913, 0.9661694216209304, 1.0292063788225723]
    # n = 58 l_max = 12
    elif N == 58:
        x = [2.681104400095342, 0.7154107707199917, 0.6789416862870981, 2.2530865495040144, 1.8673844360870282, 1.927285154465101, 1.657081274078465, 1.2191549430750221, 0.538112422091656, 1.4615780340408329, 2.9120625356827894, 1.3940910225834287, 1.1673808518901907, 1.9595090731976386, 0.5486085347557721, 1.9462168306606145, 1.0223476998510461, 2.0607184906845792, 0.8883386793240915, 1.2852150762492365, 2.895787806997049, 1.6991198381568426, 2.4510517150186244, 1.1958027838302598, 2.198845072848418, 1.6456869322125696, 2.579071228464289, 0.749948790760706, 1.0824457996934733, 1.4957407024337483, 1.7910367280457309, 1.6670247565036371, 2.23226434495121, 1.564732590741013, 1.5124900499321432, 2.1886125856021015, 1.2815538491964542, 0.36673989020787107, 2.061310567834104, 0.9749327633824536, 0.8041339533280961, 2.414340849387238, 0.3573606091636152, 2.0172009133418474, 0.7441936799638402, 1.7217704594909287, 2.4196140732903504, 1.4625732945102392, 0.19737192872794945, 1.554395366080808, 2.15571242979144, 2.6515073694563775, 1.8055944536325494, 0.9025077721183872, 1.1234157722963025, 2.4776210135817545, 1.117677748388637, 1.3440884565485327, 3.1297111908066637, 3.1093108232963274, 2.328329180993931, 3.5181797115811313, 1.3415982723199287, 5.128239532310928, 0.27416250207874604, 5.744060604676913, 1.5386371524678506, 3.7743774240100443, 4.66427341850738, 5.215498358161127, 0.06308351359734513, 2.397175382603447, 3.993892843564511, 3.966649614743391, 1.6539868090786796, 4.518646321073305, 1.0227092186157694, 3.2452205366176208, 1.517021562991158, 5.594023873585617, 2.3041965870967176, 4.730861138756998, 3.383892846989375e-08, 4.763853560130005, 0.52761351708754, 4.661672382384337, 4.18228214464703, 1.6291640801891727, 3.4021765877688606, 2.913667422738485, 0.9998465280485642, 2.0417176855383357, 2.497984534320207, 5.64678437287598, 0.6095694593202984, 5.263709270011244, 0.5029541743695377, 3.645646402536016, 0.4138427514745532, 4.997897634883493, 0.4911347637466857, 1.8546938350783773, 5.966385688898415, 0.8458413657665363, 1.5350347118450354, 6.093046430729653, 2.905130104600923, 4.295364236137228, 2.9201672205986777, 5.872128820146829, 6.062793086597214, 5.279284759766376, 2.17878262234884, 4.140574036565011, 2.749032932035534, 1.1678137651302691, 1.124421232112169, 1.1092844739094807, 1.0741579248157511, 1.010325890481436, 1.1021895932195471, 1.105697431835396, 1.1313118482827675, 1.1033833542078586, 1.0298698597963485, 1.1550923822938515, 1.0249418659908278, 1.1166940378111159, 1.0484776018269482, 1.0811059746183516, 1.0688477194155925, 1.1533787050372928, 1.1244405030765645, 1.1382441357897022, 1.0719571841657294, 1.1375498574285299, 1.0970374266721061, 1.0549545059260257, 1.1312372101111356, 1.0505329510126264, 1.0568068503592762, 0.9470789698720228, 1.0450058976907424, 0.9296589946425108, 1.0922412658398304, 0.970929891973208, 1.128379553588728, 1.079340098881142, 0.9439270000624531, 0.9610295507278086, 0.9111061032049881, 1.1169997171398396, 1.1445672952400907, 1.1318377719392254, 0.9326269243963995, 0.9966700193938546, 0.973352603331034, 1.1582105659091637, 1.060041597279727, 1.1134071700834631, 1.1587520929739463, 1.0961916606165227, 1.0118182638489503, 0.6331411223786912, 1.1450649923901937, 1.1207850392956489, 1.144807397034991, 1.0552819480135505, 0.9323747195366884, 1.1219559677355715, 1.0164378245577537, 1.1231422520847079, 1.0996791915027373, 1.1110807369508326]

    points = x_to_points(x[:2*N])
    weights = x[2*N:]  
    visualize_points(points, l_max, weights)


if __name__ == '__main__':    
    
    N = 10  # Number of points
    l_max = 2  # Maximum l value
    
    optimize_weighted(N, l_max, max_evals=20000)

    show_results(42, 10)
    show_results(48, 11)
    show_results(58, 12)
    
    
    
    
    
    