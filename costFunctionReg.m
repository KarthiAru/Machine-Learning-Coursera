function [J, grad] = costFunctionReg(theta, X, y, lambda)

	% Initialize
	m = length(y); 
	J = 0;
	grad = zeros(size(theta));

    h = 1 ./ (1+exp(-1*(X * theta)));
	J = (1/m)*sum(-y .* log(h) - (1-y) .* log(1-h)) + (lambda/(2*m))*sum(theta(2:end).^2);
	
	grad(1) = 1 / m * sum((h - y) .* X(:, 1));
	
	for i = 2:size(theta, 1)
    	grad(i) = 1 / m * sum((h - y) .* X(:, i))+ (lambda/m)*theta(i);
	end

end
