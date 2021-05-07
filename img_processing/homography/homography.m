% Calculate 3x3 homography matrix H
% Author: Manuel Serna-Aguilera

%clear all % clear workspace

% Declare our four corresponding points between two images 1 and 2.
% 1st row: x coords
% 2nd row: y coords
p1 = [0, 0, 1, 1; 0, 1, 1, 0];
p2 = [0, 0, 1, 2; 0, 1, 1, 0];

H2to1 = compute(p1, p2);

function H2to1 = compute(p1, p2)
    % Get number of points to iterate over, where n >= 4
    n = 4;
    
    % Initilize A
    A = zeros(2*n, 9);
    
    % Get x and y components from p1
    x = p1(1,:);
    y = p1(2,:);
    
    % Get u and v components from p2
    u = p2(1,:);
    v = p2(2,:);
    
    % Populate A with Ai for each point pair (xi, yi) and (ui, vi)
    for i=1:n
        A(2*i-1,:) = [x(i), y(i), 1, 0, 0, 0, x(i)*-u(i), y(i)*-u(i), -u(i)];
        A(2*i,:) = [0, 0, 0, x(i), y(i), 1, x(i)*-v(i), y(i)*-v(i), -v(i)];
    end
    
    % Take either null space or SVD of A
    if n == 4
        h = null(A);
    else
        [U, S, V] = svd(A);
        h = V(:, 9);
    end
    
    H2to1 = reshape(h, 3, 3);
end
