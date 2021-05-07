% Explore the the use of pose estimation for given 3D coords of an 
%   object/scene and their corresponding 2D positions in the image.
% Solve for: M = K[R|t]
% Author: Manuel Serna-Aguilera

% Get the rotation R and translation t, assume K is the identity matrix
function [R, t] = estimateCamera(xy, XYZ)
    % Set number of points
    n = 23;
    
    % Initialize A as (2*n)x12 matrix
    A = zeros(2*n, 12);
    
    % Get points from xy (denoted by lower-case x and y coord vectors)
    x = xy(:, 1);
    y = xy(:, 2);
    
    % Get points from XYZ (denoted by upper-case X, Y, Z coord vectors)
    X = XYZ(:, 1);
    Y = XYZ(:, 2);
    Z = XYZ(:, 3);
    
    % Populate A
    for i=1:n
        A(2*i-1,:) = [X(i), Y(i), Z(i), 1, 0, 0, 0, 0, -x(i)*X(i), -x(i)*Y(i), -x(i)*Z(i), -x(i)];
        A(2*i,:) = [0, 0, 0, 0, X(i), Y(i), Z(i), 1, -y(i)*X(i), -y(i)*Y(i), -y(i)*Z(i), -y(i)];
    end
    
    % Solve for camera matrix M by SVD
    [U, S, V] = svd(A);
    N = V(:, 12);
    M = reshape(N, 3, 4);
    
    % Now, separate R and t in M=[R|t]
    R = M(:, 1:3);
    t = M(:, 4);
end
    
