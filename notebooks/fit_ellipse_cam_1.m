code = '/Users/yushi/OneDrive/Academic/UoB/BCFN/BCFN-Projects/Fish/code';
repo = '/Guaranteed Ellipse Fitting with a Confidence Region';
addpath(strcat(code, repo))
addpath('Fast Guaranteed AML Ellipse Fit with Uncertainty');
addpath('Direct Ellipse Fit');
addpath('Conversion Functions');
addpath('Covariance');
addpath('Helper Functions');

mySeed = 10; 
rand('seed', mySeed );
randn('seed', mySeed );

data_points = [
    8.000,333.000;
    39.000,295.000;
    70.000,266.000;
    97.000,240.000;
    121.000,217.000;
    149.000,195.000;
    177.000,173.000;
    203.000,154.000;
    228.000,135.000;
    261.000,116.000;
    290.000,98.000;
    311.000,86.000;
    335.000,72.000;
    363.000,59.333;
    389.667,44.667;
    413.000,35.000;
    439.000,24.000;
    461.000,14.000;
    485.000,5.000;
    1531.333,18.667;
    1570.000,33.333;
    1616.000,60.000;
    1653.000,81.000;
    1689.000,106.000;
    1721.000,127.000;
    1760.000,154.000;
    1790.000,179.000;
    1809.667,194.333;
    1835.667,222.333;
    1855.667,245.000;
    1876.333,267.000;
    1895.667,291.000;
    1913.000,315.000;
    1930.333,340.333;
    1943.667,362.000;
    1956.333,384.000;
    1965.667,404.667;
    1973.000,425.333;
    1980.333,447.333;
    1989.667,472.000;
    1998.333,499.333;
    1999.667,518.000;
    2003.667,542.667;
    2005.667,564.000;
    2007.000,588.667;
    2008.333,606.000;
    2006.000,633.000;
    2005.000,652.000;
    2005.667,675.333;
    2000.333,697.333;
    1995.000,722.000;
    1989.667,740.000;
    1983.667,762.667;
    1974.000,789.000;
    1958.000,824.000;
    1943.000,859.000;
    1923.000,897.000;
    1900.000,930.000;
    1862.000,981.000;
    1884.000,957.000;
    1840.000,1014.000;
    1817.000,1038.000;
    1798.000,1050.000;
    1778.000,1066.333;
    1752.000,1087.000;
    1717.000,1117.000;
    1687.000,1138.000;
    1647.000,1163.000;
    1609.000,1185.000;
    1571.000,1210.000;
    1527.000,1229.000;
    1479.000,1252.000;
    1435.000,1271.000;
    1380.000,1291.000;
    1329.000,1308.000;
    1288.000,1317.000;
    1245.000,1330.000;
    1196.000,1337.000;
    1146.000,1348.000;
    1079.000,1355.000;
    1010.000,1364.000;
    895.000,1366.333;
    845.667,1362.333;
    797.667,1359.667;
    745.667,1354.333;
    699.000,1349.000;
    960.333,1365.000;
    634.000,1338.333;
    586.000,1329.000;
    554.000,1321.000;
    495.333,1305.000;
    460.667,1293.000;
    410.000,1274.333;
    364.667,1255.667;
    312.000,1238.000;
    262.667,1206.333;
    216.000,1178.333;
    169.333,1147.667;
    113.333,1103.667;
    72.000,1062.333;
    49.333,1038.333;
    21.333,1009.000;
    4.000,981.000;
];

% An example of fitting to all the data points
fprintf('**************************************************************\n')
fprintf('* Example with ALL data points assuming homogeneous Gaussian *\n')
fprintf('* noise with the noise level automatically estimated from    *\n')
fprintf('* data points                                                *\n')
fprintf('**************************************************************\n')

fprintf('Algebraic ellipse parameters of direct ellipse fit: \n')
[theta_dir]  = compute_directellipse_estimates(data_points)
fprintf('Algebraic ellipse parameters of our method: \n')
[theta_fastguaranteed] = fast_guaranteed_ellipse_estimate(data_points)


fprintf('Geometric ellipse parameters \n')
fprintf('(majAxis, minAxis, xCenter,yCenter, orientation (radians)): \n')
geometricEllipseParameters = ...
            fromAlgebraicToGeometricParameters(theta_fastguaranteed)

fprintf('Covariance matrix of geometric parameters: \n')
geoCov =  compute_covariance_of_geometric_parameters(...
                               theta_fastguaranteed, data_points)
 
 fprintf('Standard deviation of geometric parameters: \n')                          
 stds = sqrt(diag(geoCov)) 
 
 
[S, thetaCovarianceMatrixNormalisedSpace] = ...
                                 compute_covariance_of_sampson_estimate(...
                     theta_fastguaranteed, data_points);
   
% plot the data points
x = data_points;
n = length(x);
figure('Color',[1 1 1])

s = scatter(x(:,1), x(:,2), 'k.');

minX = 0;
maxX = 2000;
minY = 0;
maxY = 1500;

% plot the direct ellipse fit
a = theta_dir(1); b = theta_dir(2); c = theta_dir(3);
d = theta_dir(4); e = theta_dir(5); f = theta_dir(6);
fh = @(x,y) (a*x.^2 + b*x.*y + c*y.^2 + d*x + e*y + f);
h = ezplot(fh,[minX maxX minY maxY]);


% plot the guaranteed ellipse fit
a = theta_fastguaranteed(1); b = theta_fastguaranteed(2);
c = theta_fastguaranteed(3); d = theta_fastguaranteed(4); 
e = theta_fastguaranteed(5); f = theta_fastguaranteed(6);
fh = @(x,y) (a*x.^2 + b*x.*y + c*y.^2 + d*x + e*y + f);
h = ezplot(fh,[minX maxX minY maxY]);