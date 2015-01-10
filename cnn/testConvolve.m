a = reshape(randn(6), 6, [])
b = reshape(randn(3), 3, 3)
e = rot90(b, 2);
c = conv2(a, e, 'valid');
c = 1 ./ (1 + exp(-c));
d = zeros(4, 4);
for i = 1 : 4
	for j = 1 : 4
		d(i, j) = sum(sum(b .* a(i : i + 2, j : j + 2)));
		d(i, j) = 1 ./ (1 + exp(-d(i, j)));
	end
end
c
d
