#include <iostream>
#include <random>
#include <vector>
#include <variant>
#include <functional>
#include <string>
#include <numeric>
#include <chrono>
#include <malloc.h>
#include <cassert>
#include <type_traits>
#include <complex>
#include <cassert>
#include <cmath>

#include <windows.h>
#ifdef __AVX__
#include <immintrin.h>
#endif

#define max(a,b) (((a) > (b)) ? (a) : (b))
#define min(a,b) (((a) < (b)) ? (a) : (b))

using namespace std;

const double PI = 3.14159265358979323846;
// Déclarer le générateur de nombres aléatoires basé sur l'algorithme Mersenne Twister
static std::mt19937 gen(std::random_device{}());

template <typename NUMBER_TYPE>
static std::vector<NUMBER_TYPE> range(NUMBER_TYPE x) {
	std::vector<NUMBER_TYPE> numbers(x);
	std::iota(numbers.begin(), numbers.end(), 0);
	return numbers;
};

template <typename NUMBER_TYPE>
static std::vector<NUMBER_TYPE> range(NUMBER_TYPE x, NUMBER_TYPE y) {
	std::vector<NUMBER_TYPE> numbers(y - x + 1);
	std::iota(numbers.begin(), numbers.end(), x);
	return numbers;
};

template <typename NUMBER_TYPE>
static std::vector<NUMBER_TYPE> range(NUMBER_TYPE x, NUMBER_TYPE y, NUMBER_TYPE step) {
	std::vector<NUMBER_TYPE> numbers((y - x) / step);
	if (step > 0) {
		for (NUMBER_TYPE i = x; i < y; i += step)
		{
			numbers.push_back(i);
		}
	}
	else if (step < 0)
	{
		for (NUMBER_TYPE i = x; i > y; i += step)
		{
			numbers.push_back(i);
		}
	}
	else
	{
		throw std::invalid_argument("step cannot be zero");
	}
	return numbers;
};

template <typename T>
static void showVector(std::vector<T> res, std::string separator = ", ", bool endLine = true) {
	if (res.empty()) { // Si le vecteur est vide, on affiche rien
		return;
	}
	std::cout << res[0];
	res.erase(res.begin());
	for (const T& x : res) {
		std::cout << separator << x;
	}
	if (endLine)
	{
		std::cout << std::endl;
	}
}

template <typename NUMBER_TYPE = int>
	requires std::is_arithmetic_v<NUMBER_TYPE>
static NUMBER_TYPE randomGaussian(NUMBER_TYPE mean = 0, NUMBER_TYPE stddev = 1) {
	// Créer un générateur aléatoire basé sur l'algorithme Mersenne Twister
	static std::mt19937 gen(std::random_device{}());
	// Créer une distribution normale avec la moyenne et l'écart-type donnés
	std::normal_distribution<NUMBER_TYPE> dist(mean, stddev);
	// Générer et renvoyer un nombre aléatoire selon la distribution
	return dist(gen);
}

static float uniformGaussian()
{
	// Créer une distribution uniforme entre -1 et 1
	std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
	// Générer et renvoyer un nombre aléatoire selon la distribution
	return dist(gen);
}

template <typename NUMBER_TYPE = int>
	requires std::is_arithmetic_v<NUMBER_TYPE>
static NUMBER_TYPE randomUniform(NUMBER_TYPE a, NUMBER_TYPE b)
{
	static std::random_device rd; // générateur de nombres aléatoires
	static std::mt19937 gen(rd()); // moteur de nombres aléatoires
	std::uniform_real_distribution<> dis(a, b); // distribution uniforme entre a et b
	return NUMBER_TYPE(dis(gen)); // nombre aléatoire entre a et b
}

template <typename NUMBER_TYPE = int>
	requires std::is_arithmetic_v<NUMBER_TYPE>
static NUMBER_TYPE randomNumber()
{
	std::random_device rd;
	std::mt19937 g(rd());
	return g();
}

// Convertit un vecteur multi-dimensionnel en un vecteur 1D
template <typename T>
vector<T> flatten(vector<T> v)
{
	// Cas de base : le vecteur est déjà 1D, on le retourne tel quel
	return v;
}

// Convertit un vecteur multi-dimensionnel en un vecteur 1D
template <typename T, typename... Args>
vector<T> flatten(vector<vector<Args...>> v)
{
	// Cas récursif : le vecteur est au moins 2D, on applique flatten à chaque sous-vecteur
	vector<T> result;
	for (auto& subv : v)
	{
		// On concatène le résultat de flatten(subv) au vecteur résultat
		vector<T> temp = flatten<T>(subv);
		result.insert(result.end(), temp.begin(), temp.end());
	}
	return result;
}

static int getIndex(vector<int> indexs, vector<int> maxi)
{
	int index = 0;
	int factor = 1;
	int size = static_cast<int>(maxi.size());
	if (size != static_cast<int>(indexs.size())) throw runtime_error("You need to give the name shape of indexs");
	for (int i = size - 1; i >= 0; i--)
	{
		index += indexs[i] * factor;
		factor *= maxi[i];
	}
	return index;
}

static vector<int> getCoords(int index, const vector<int>& shape) {
	vector<int> coords(shape.size());
	for (int i = shape.size() - 1; i >= 0; i--) {
		coords[i] = index % shape[i];
		index /= shape[i];
	}
	return coords;
}

static void incrementIndex(vector<int>& index, vector<int> shape)
{
	int carry = 1;
	for (int i = static_cast<int>(index.size()) - 1; i >= 0; i--)
	{
		index[i] += carry;
		if (index[i] >= shape[i])
		{
			index[i] = 0;
			carry = 1;
		}
		else
		{
			carry = 0;
		}
	}
}

template <typename NUMBER_TYPE = int>
	requires std::is_arithmetic_v<NUMBER_TYPE>
static vector<NUMBER_TYPE> squeeze(std::vector<NUMBER_TYPE> v)
{
	vector<NUMBER_TYPE> result;
	for (int i = 0; i < static_cast<int>(v.size()); i++) {
		if (v[i] > 1) {
			result.push_back(v[i]);
		}
	}
	return result;
}

template <typename T>
static void shuffle(std::vector<T>& v) {
	std::random_device rd;
	std::mt19937 g(rd());
	std::shuffle(v.begin(), v.end(), g);
}

template <typename T>
static vector<T> slice(vector<T>& v, int x, int y) {
	if (x < 0 || x >= v.size() || y < 0 || y > v.size() - x) throw invalid_argument("Paramètres invalides pour la fonction couper");
	vector<T> resultat;
	resultat.reserve(y);
	for (int i = x; i < x + y; i++) {
		resultat.push_back(v[i]);
	}
	return resultat;
}

vector<int> convolve_output_shape(int output_dim, int filter_num, vector<int> input, vector<int> kernel, vector<int> strides = {}, vector<int> dilatations = {}, const string padding = "valid")
{
	input.resize(output_dim, 1);
	kernel.resize(output_dim + 1, 1);
	strides.resize(output_dim, 1);
	dilatations.resize(output_dim, 1);
	vector<int> output_shape = { filter_num };
	if (padding == "valid") {
		for (int i = 0; i < output_dim; i++) {
			output_shape.push_back((input[i] - kernel[i + 1] - (kernel[i + 1] - 1) * (dilatations[i] - 1)) / strides[i] + 1);
		}
	}
	else if (padding == "same") {
		for (int i = 0; i < output_dim; i++) {
			output_shape.push_back((input[i] - strides[i]) / strides[i] + 1);
		}
	}
	else if (padding == "full") {
		for (int i = 0; i < output_dim; i++) {
			output_shape.push_back((input[i] + 2 * (kernel[i + 1] - 1) * dilatations[i] + strides[i] - 2) / strides[i]);
		}
	}
	return output_shape;
}

// Cette fonction calcule la WHT d'un vecteur de données en utilisant l'algorithme fwht
// Elle modifie le vecteur sur place
template <typename NUMBER_TYPE>
	requires std::is_arithmetic_v<NUMBER_TYPE> || std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value
static void fwht(vector<NUMBER_TYPE>& data, int n) {
	// n doit être une puissance de 2
	if (n == 0 || (n & (n - 1)) != 0) {
		throw runtime_error("La taille du vecteur doit être une puissance de 2");
	}
	// On applique l'algorithme fwht récursivement
	for (int len = 1; len < n; len *= 2) {
		for (int i = 0; i < n; i += 2 * len) {
			for (int j = 0; j < len; j++) {
				NUMBER_TYPE u = data[i + j];
				NUMBER_TYPE v = data[i + j + len];
				data[i + j] = u + v;
				data[i + j + len] = u - v;
			}
		}
	}
}

// Cette fonction calcule la WHT inverse d'un vecteur de données en utilisant l'algorithme fwht
// Elle modifie le vecteur sur place
template <typename NUMBER_TYPE>
	requires std::is_arithmetic_v<NUMBER_TYPE> || std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value
static void ifwht(vector<NUMBER_TYPE>& data, int n) {
	// n doit être une puissance de 2
	if (n == 0 || (n & (n - 1)) != 0) {
		throw runtime_error("La taille du vecteur doit être une puissance de 2");
	}
	// On applique la WHT directe
	fwht(data, n);
	// On divise chaque élément par n
	for (int i = 0; i < n; i++) {
		data[i] /= n;
	}
}

template <typename NUMBER_TYPE>
	requires std::is_arithmetic_v<NUMBER_TYPE>
class FFT {
public:
	FFT(int size) {
		size_ = size;
		log2n_ = std::log2(size);
		coefficients_.resize(size);
	}
	std::vector<std::complex<NUMBER_TYPE>> transform(std::vector<std::complex<NUMBER_TYPE>> input) {
		for (int i = 0; i < size_; i++) {
			coefficients_[i] = input[i];
		}
		fft(coefficients_, log2n_);
		return coefficients_;
	}
	std::vector<std::complex<NUMBER_TYPE>> inverse_transform(std::vector<std::complex<NUMBER_TYPE>> input) {
		for (int i = 0; i < size_; i++) {
			coefficients_[i] = input[i];
		}
		fft(coefficients_, log2n_, true);
		for (int i = 0; i < size_; i++) {
			coefficients_[i] /= size_;
		}
		return coefficients_;
	}

private:
	void fft(std::vector<std::complex<NUMBER_TYPE>>& x, int log2n, bool inverse = false) {
		int n = 1 << log2n;

		for (int i = 0; i < n; i++) {
			int j = 0;
			for (int k = 0; k < log2n; k++) {
				j = (j << 1) | ((i >> k) & 1);
			}
			if (j > i) {
				std::swap(x[i], x[j]);
			}
		}
		for (int s = 1; s <= log2n; s++) {
			int m = 1 << s;
			double sign = inverse ? 1 : -1;
			std::complex<NUMBER_TYPE> wm = std::polar(NUMBER_TYPE(1.0), NUMBER_TYPE(sign * 2.0 * PI / m));
			for (int k = 0; k < n; k += m) {
				std::complex<NUMBER_TYPE> w = 1.0;
				for (int j = 0; j < m / 2; j++) {
					std::complex<NUMBER_TYPE> t = w * x[k + j + m / 2];
					std::complex<NUMBER_TYPE> u = x[k + j];
					x[k + j] = u + t;
					x[k + j + m / 2] = u - t;
					w *= wm;
				}
			}
		}
	}
	int size_;
	int log2n_;
	std::vector<std::complex<NUMBER_TYPE>> coefficients_;
};

template <typename NUMBER_TYPE = double>
	requires std::is_arithmetic_v<NUMBER_TYPE> || std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value
class Tensor {
public:
	vector<NUMBER_TYPE> data;
	vector<int> shape;
	int length;
	int dimension;

	Tensor(const vector<int>& shape = { 1 })
	{
		this->resize(shape);
	}

	Tensor(const vector<NUMBER_TYPE>& data, const vector<int>& shape)
	{
		this->data = data;
		this->resize(shape);
	}

	Tensor(vector<NUMBER_TYPE>& data, vector<int>& shape, NUMBER_TYPE(*function)(NUMBER_TYPE))
	{
		this->data = data;
		this->resize(shape);
		if (shape.size() < 0) throw runtime_error("Shape of Tensor must be more than 1 and in a vector");
		this->apply(function);
	}

	NUMBER_TYPE maximum()
	{
		return *std::max_element(this->data.begin(), this->data.end());
	}

	NUMBER_TYPE minimum()
	{
		return *std::min_element(this->data.begin(), this->data.end());
	}

	NUMBER_TYPE get(vector<int> indexs)
	{
		int index = getIndex(indexs, this->shape);
		return data[index];
	}

	void set(vector<int> indexs, NUMBER_TYPE value)
	{
		int index = getIndex(indexs, this->shape);
		data[index] = value;
	}

	auto real()
	{
		assert((std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value));
		using COMPLEX_TYPE = NUMBER_TYPE::value_type;
		Tensor<COMPLEX_TYPE> result(this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [](NUMBER_TYPE c) { return c.real(); });
		return result;
	}

	auto imag()
	{
		assert((std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value));
		using COMPLEX_TYPE = NUMBER_TYPE::value_type;
		Tensor<COMPLEX_TYPE> result(this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [](NUMBER_TYPE c) { return c.imag(); });
		return result;
	}

	auto abs()
	{
		using COMPLEX_TYPE = NUMBER_TYPE::value_type;
		Tensor<COMPLEX_TYPE> result(this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [](NUMBER_TYPE c) { return c.abs(); });
		return result;
	}

	Tensor<NUMBER_TYPE> getMult(vector<int> indices)
	{
		int size = static_cast<int>(indices.size());
		if (size >= static_cast<int>(this->dimension)) throw runtime_error("You cannot get multiple values if you don't use less indices than the shape");
		vector<int> subShape(this->shape.begin() + size, this->shape.end());
		int subSize = 1;
		for (int x : subShape)
		{
			subSize *= x;
		}
		int index = getIndex(indices, ::slice(this->shape, 0, size));
		vector<NUMBER_TYPE> subData(this->data.begin() + index, this->data.begin() + index + subSize);
		return Tensor<NUMBER_TYPE>(subData, subShape);
	}

	void setMult(vector<int> indices, Tensor<NUMBER_TYPE> value)
	{
		int size = static_cast<int>(indices.size());
		if (size >= static_cast<int>(this->dimension)) throw runtime_error("You cannot set multiple values if you don't use less indices than the shape");
		vector<int> subShape(this->shape.begin() + size, this->shape.end());
		if (subShape != value.shape) throw runtime_error("You cannot set multiple values with a different shape");
		int subSize = 1;
		for (int x : subShape)
		{
			subSize *= x;
		}
		int index = getIndex(indices, ::slice(this->shape, 0, size));
		std::copy(value.data.begin(), value.data.end(), this->data.begin() + index);
	}

	Tensor<NUMBER_TYPE> slice(const vector<int>& start, const vector<int>& end)
	{
		if (start.size() != this->dimension || end.size() != this->dimension) throw runtime_error("Les vecteurs start et end doivent avoir la même taille que la dimension du tenseur");
		vector<int> subShape(this->dimension);
		for (int i = 0; i < this->dimension; i++)
		{
			if (start[i] < 0 || start[i] >= this->shape[i] || end[i] < start[i] || end[i] > this->shape[i]) throw runtime_error("Les indices de début et de fin doivent être valides pour chaque dimension");
			subShape[i] = end[i] - start[i];
		}
		vector<NUMBER_TYPE> subData;
		for (int i = 0; i < this->length; i++)
		{
			vector<int> indices = getCoords(i, this->shape);
			bool inSubTensor = true;
			for (int j = 0; j < this->dimension; j++)
			{
				if (indices[j] < start[j] || indices[j] >= end[j])
				{
					inSubTensor = false;
					break;
				}
			}
			if (inSubTensor) subData.push_back(this->data[i]);
		}
		Tensor<NUMBER_TYPE> subTensor(subData, subShape);
		return subTensor;
	}

	void flatten()
	{
		this->shape = { this->length };
		this->dimension = 1;
	}

	Tensor<NUMBER_TYPE> copy()
	{
		Tensor<NUMBER_TYPE> result(this->shape);
		result.data = this->data;
		result.length = this->length;
		result.dimension = this->dimension;
		return result;
	}

	Tensor<NUMBER_TYPE> apply(NUMBER_TYPE(*function)(...))
	{
		Tensor<NUMBER_TYPE> result = this->copy();
		transform(result.data.begin(), result.data.end(), result.data.begin(), function);
		return result;
	}

	Tensor<NUMBER_TYPE> apply(std::function<NUMBER_TYPE(NUMBER_TYPE)> f)
	{
		Tensor<NUMBER_TYPE> result = this->copy();
		for (size_t i = 0; i < result.data.size(); i++) {
			result.data[i] = f(result.data[i]); // Appliquer la fonction f à chaque élément du vecteur data
		}
		return result;
	}

	void fill(NUMBER_TYPE value, bool end_only = false)
	{
		if (end_only)
		{
			auto start = std::find_if(this->data.rbegin(), this->data.rend(), [](NUMBER_TYPE x){ return x != 0; }).base();
			std::transform(start, this->data.end(), start, [value]{ return value; });
		}
		else std::transform(this->data.begin(), this->data.end(), this->data.begin(), [value](NUMBER_TYPE x) { return value; });
	}

	void fill(std::function<NUMBER_TYPE(int, const vector<int>&)> f, bool end_only = false)
	{
		if (end_only)
		{
			auto start = std::find_if(this->data.rbegin(), this->data.rend(), [](NUMBER_TYPE x) { return x != 0; }).base();
			int index = std::distance(this->data.begin(), start);
			std::transform(start, this->data.end(), start, this, &index, [f,index](NUMBER_TYPE x) {
				NUMBER_TYPE y = f(x,index, this->shape);
				index++;
				return y;
				});
		}
		else
		{
			int index = 0;
			std::transform(this->data.begin(), this->data.end(), this->data.begin(), this, &index, [f, index](NUMBER_TYPE x) {
				NUMBER_TYPE y = f(x, index, this->shape);
				index++;
				return y;
				});
		}
	}

	void resize(vector<int> newShape) {
		int newSize = 1;
		for (int x : newShape) {
			newSize *= x;
		}
		this->data.reserve(newSize);
		this->data.resize(newSize);
		this->data.reserve(newShape.size());
		this->shape = newShape;
		this->dimension = shape.size();
		this->length = data.size();
	}

	void pad(vector<int> axes, vector<int> amounts, vector<bool> sides = {})
	{
		if (axes.size() != amounts.size() || (sides.size() != 0 && sides.size() != axes.size()))
		{
			throw runtime_error("Invalid parameters for pad method");
		}

		vector<int> new_shape = this->shape;
		for (int i = 0; i < axes.size(); i++)
		{
			int axis = axes[i];
			int amount = amounts[i];
			bool side = sides.size() == 0 ? false : sides[i];
			if (axis < 0 || axis >= this->dimension)
			{
				throw runtime_error("Invalid axis for pad method");
			}
			if (amount < 0)
			{
				if (-amount > new_shape[axis]) throw runtime_error("Cannot remove more elements than existing");
				new_shape[axis] += amount;
			}
			else new_shape[axis] += amount;
		}

		vector<int> index(this->dimension, 0);
		for (int i = 0; i < length; i++)
		{
			int dataIndex = getIndex(index, shape);
			int newDataIndex = 0;
			int stride = 1;
			for (int j = dimension - 1; j >= 0; j--)
			{
				int axis = j;
				int amount = 0;
				bool side = false;
				for (int k = 0; k < axes.size(); k++)
				{
					if (axes[k] == j)
					{
						amount = amounts[k];
						side = sides.size() == 0 ? false : sides[k];
						break;
					}
				}
				if (amount < 0)
				{
					if (side) newDataIndex += (index[j] + amount) * stride;
					else newDataIndex += index[j] * stride;
				}
				else
				{
					if (side) newDataIndex += (index[j] + amount) * stride;
					else newDataIndex += index[j] * stride;
				}
				stride *= new_shape[j];
			}
			if (newDataIndex > dataIndex) data.insert(data.begin() + dataIndex, newDataIndex - dataIndex, 0);
			int paddingAfter = 0;
			for (int k = 0; k < axes.size(); k++)
			{
				if (axes[k] == dimension - 1)
				{
					paddingAfter = amounts[k];
					break;
				}
			}
			if (paddingAfter > 0) data.insert(data.begin() + newDataIndex + 1, paddingAfter, 0);
			incrementIndex(index, shape);
		}
		shape = new_shape;
		dimension = shape.size();
		length = data.size();
	}

	void shift(vector<int> axes, vector<int> positions, vector<int> amounts)
	{
		// Vérifier que les paramètres sont valides
		if (axes.size() != positions.size() || axes.size() != amounts.size())
		{
			throw runtime_error("Invalid parameters for shift method");
		}

		// Créer un nouveau tensor avec la même forme que le tensor original
		Tensor<NUMBER_TYPE> result(this->shape);

		// Copier les données du tensor original dans le nouveau tensor en les décalant
		vector<int> index(this->dimension, 0);
		bool done = false;
		while (!done)
		{
			// Calculer l'index linéaire dans le tensor original
			int old_index = 0;
			int old_stride = 1;
			for (int i = this->dimension - 1; i >= 0; i--)
			{
				old_index += index[i] * old_stride;
				old_stride *= this->shape[i];
			}

			// Calculer l'index linéaire dans le nouveau tensor
			int new_index = 0;
			int new_stride = 1;
			for (int i = axes.size() - 1; i >= 0; i--)
			{
				int axis = axes[i];
				int position = positions[i];
				int amount = amounts[i];
				// Ajuster l'index selon le paramètre
				if (index[axis] >= position)
				{
					// Décaler les éléments à partir de la position indiquée
					new_index += (index[axis] + amount) * new_stride;
				}
				else
				{
					// Garder les éléments avant la position indiquée
					new_index += index[axis] * new_stride;
				}
				new_stride *= this->shape[axis];
			}

			// Copier la valeur du tensor original dans le nouveau tensor si l'index est valide
			if (new_index >= 0 && new_index < this->length)
			{
				result.data[new_index] = this->data[old_index];
			}

			// Passer à l'index suivant
			int carry = 1;
			for (int i = this->dimension - 1; i >= 0; i--)
			{
				index[i] += carry;
				if (index[i] >= this->shape[i])
				{
					index[i] = 0;
					carry = 1;
				}
				else
				{
					carry = 0;
					break;
				}
			}
			if (carry == 1)
			{
				done = true;
			}
		}

		// Remplacer le tensor original par le nouveau tensor
		this->data = result.data;
		this->shape = result.shape;
		this->length = result.length;
		this->dimension = result.dimension;
	}

	void squeeze()
	{
		vector<int> new_shape = this->shape;
		::squeeze(new_shape);
		this->shape = new_shape;
		this->dimension = static_cast<int>(new_shape.size());
	}

	void reset()
	{
		this->data.assign(this->data.size(), 0);
		this->resize(shape);
	}

	NUMBER_TYPE sum()
	{
		return std::reduce(this->data.begin(), this->data.end());
	}

	NUMBER_TYPE product()
	{
		return std::accumulate(this->data.begin(), this->data.end(), 1, std::multiplies<NUMBER_TYPE>());
	}

	void randomize()
	{
		// Créer une distribution normale avec la moyenne et l'écart-type par défaut
		std::normal_distribution<NUMBER_TYPE> dist;
		// Remplir le vecteur data avec des nombres aléatoires selon la distribution
		std::generate(data.begin(), data.end(), [&] { return dist(gen); });
	}

	void shuffle() {
		::shuffle(this->data);
	}

	Tensor<NUMBER_TYPE> transpose(vector<int> order = {}) {
		if (order.empty())
		{
			order.resize(this->dimension);
			std::iota(order.begin(), order.end(), 0);
			std::reverse(order.begin(), order.end());
		}
		if (order.size() != this->dimension) throw runtime_error("The order vector must have the same size as the dimension of the tensor");
		vector<int> newShape(this->dimension);
		for (int i = 0; i < this->dimension; i++) newShape[i] = this->shape[order[i]];
		Tensor<NUMBER_TYPE> result(newShape);
		for (int i = 0; i < this->length; i++)
		{
			vector<int> index = getIndices(i, this->shape);
			vector<int> newIndex(this->dimension);
			for (int j = 0; j < this->dimension; j++) newIndex[j] = index[order[j]];
			result.set(newIndex, this->get(index));
		}
		std::swap(this->shape, result.shape);
		this->data = result.data;
		return *this;
	}

	void show()
	{
		int dim = static_cast<int>(dimension);
		if (dim == 1)
		{
			cout << "{ ";
			showVector(data, ", ", false);
			cout << " }";
		}
		else
		{
			cout << "{" << endl;
			int subSize = this->length / shape[0];
			vector<int> subShape(shape.begin() + 1, shape.end());
			for (int i = 0; i < shape[0]; i++)
			{
				cout << "  ";
				vector<NUMBER_TYPE> subData(data.begin() + i * subSize, data.begin() + (i + 1) * subSize);
				Tensor<NUMBER_TYPE> subTensor(subData, subShape);
				subTensor.show();
				if (i < shape[0] - 1)
				{
					cout << ",";
				}
				cout << endl;
			}
			cout << "} ";
		}
	}

	Tensor<NUMBER_TYPE> reduce(std::function<NUMBER_TYPE(NUMBER_TYPE, NUMBER_TYPE)> op, int axis)
	{
		if (axis < -1 || axis >= dimension)
		{
			throw std::invalid_argument("Invalid axis");
		}

		if (axis == -1)
		{
			NUMBER_TYPE result = data[0];
			for (int i = 1; i < length; i++)
			{
				result = op(result, data[i]);
			}
			return Tensor<NUMBER_TYPE>({ result }, { 1 });
		}

		vector<int> new_shape = shape;
		int new_length = length / shape[axis];
		new_shape.erase(new_shape.begin() + axis);
		vector<NUMBER_TYPE> new_data(new_length);

		for (int i = 0; i < length; i++)
		{
			int new_index = 0;
			int factor = 1;
			for (int j = dimension - 1; j >= 0; j--)
			{
				if (j != axis)
				{
					new_index += (i / factor) % shape[j] * factor;
					factor *= shape[j];
				}
			}
			new_data[new_index] = op(new_data[new_index], data[i]);
		}
		return Tensor<NUMBER_TYPE>(new_data, new_shape);
	}

	Tensor<NUMBER_TYPE> pow(NUMBER_TYPE x)
	{
		Tensor<NUMBER_TYPE> result(this->shape);
		for (int i = 0; i < this->data.size(); i++)
		{
			result.data[i] = std::pow(this->data[i], x);
		}
		return result;
	}

	Tensor<NUMBER_TYPE> sqrt()
	{
		Tensor<NUMBER_TYPE> result(this->shape);
		for (int i = 0; i < this->data.size(); i++)
		{
			result.data[i] = std::sqrt(this->data[i]);
		}
		return result;
	}

	double norm2()
	{
		double sum = 0.0;
		for (int i = 0; i < this->length; i++)
		{
			sum += std::pow(this->data[i], 2);
		}
		return std::sqrt(sum);
	}

	Tensor<NUMBER_TYPE> normalize() {
		// On copie les données du tenseur dans un vecteur
		vector<NUMBER_TYPE> data = this->data;
		// On calcule la norme du vecteur
		NUMBER_TYPE norm = 0;
		for (int i = 0; i < data.size(); i++) {
			norm += data[i] * data[i];
		}
		norm = std::sqrt(norm);
		// On divise le vecteur par la norme
		for (int i = 0; i < data.size(); i++) {
			data[i] /= norm;
		}
		// On renvoie un nouveau tenseur avec les données normalisées
		return Tensor<NUMBER_TYPE>(data, this->shape);
	}

	Tensor<NUMBER_TYPE> permute(vector<int> indices) {
		// copier le tenseur
		Tensor<NUMBER_TYPE> result = this->copy();
		// vérifier que le vecteur d'indices est valide
		if (indices.size() != this->dimension) {
			throw runtime_error("Le vecteur d'indices doit avoir la même taille que la dimension du tenseur");
		}
		vector<bool> seen(this->dimension, false); // un vecteur pour marquer les indices vus
		for (int i = 0; i < this->dimension; i++) {
			int index = indices[i];
			if (index < 0 || index >= this->dimension || seen[index]) {
				throw runtime_error("Le vecteur d'indices doit contenir tous les indices de 0 à dimension-1 sans répétition");
			}
			seen[index] = true; // marquer l'indice comme vu
		}
		// permuter les dimensions du tenseur selon le vecteur d'indices
		vector<int> new_shape(this->dimension); // la nouvelle forme du tenseur
		vector<NUMBER_TYPE> new_data(this->length); // les nouvelles données du tenseur
		vector<int> old_index(this->dimension); // l'index dans l'ancien tenseur
		vector<int> new_index(this->dimension); // l'index dans le nouveau tenseur
		for (int i = 0; i < this->length; i++) {
			// calculer l'index dans l'ancien tenseur
			old_index = vector<int>(this->dimension, 0);
			incrementIndex(old_index, this->shape);
			// calculer l'index dans le nouveau tenseur
			for (int j = 0; j < this->dimension; j++) {
				new_index[j] = old_index[indices[j]];
				new_shape[j] = this->shape[indices[j]];
			}
			// copier la valeur du tenseur à la nouvelle position
			new_data[getIndex(new_index, new_shape)] = this->data[getIndex(old_index, this->shape)];
		}
		// mettre à jour le tenseur avec les nouvelles données et la nouvelle forme
		result.data = new_data;
		result.shape = new_shape;
		// retourner le résultat
		return result;
	}

	Tensor<NUMBER_TYPE> matmul(const Tensor<NUMBER_TYPE>& other) {
		int d1 = this->dimension;
		int d2 = other.dimension;
		Tensor<NUMBER_TYPE> A = *this;
		Tensor<NUMBER_TYPE> B = other;
		if (d1 == 1) {
			A.resize({ 1, A.shape[0] });
			d1 = 2;
		}
		if (d2 == 1) {
			B.resize({ B.shape[0], 1 });
			d2 = 2;
		}
		if (d1 != 2 || d2 != 2 || A.shape[1] != B.shape[0]) {
			throw runtime_error("You cannot multiply those tensors");
		}
		int m = A.shape[0];
		int n = A.shape[1];
		int p = B.shape[1];
		Tensor<NUMBER_TYPE> result({ m, p });
#ifdef __AVX__
		if constexpr (std::is_integral<NUMBER_TYPE>::value) {
			if constexpr (sizeof(NUMBER_TYPE) == 4) {
				for (int i = 0; i < m; i++) {
					for (int j = 0; j < p; j += 8) {
						__m256i sum = _mm256_setzero_si256();
						for (int k = 0; k < n; k++) {
							__m256i a = _mm256_set1_epi32(A.data[i * n + k]);
							__m256i b = _mm256_loadu_si256((__m256i*) & B.data[k * p + j]);
							sum = _mm256_add_epi32(sum, _mm256_mullo_epi32(a, b));
						}
						_mm256_storeu_si256((__m256i*) & result.data[i * p + j], sum);
					}
				}
			}
			else if constexpr (sizeof(NUMBER_TYPE) == 8) {
				for (int i = 0; i < m; i++) {
					for (int j = 0; j < p; j += 4) {
						__m256i sum = _mm256_setzero_si256();
						for (int k = 0; k < n; k++) {
							__m256i a = _mm256_set1_epi64x(A.data[i * n + k]);
							__m256i b = _mm256_loadu_si256((__m256i*) & B.data[k * p + j]);
							sum = _mm256_add_epi64(sum, _mm256_mullo_epi64(a, b));
						}
						_mm256_storeu_si256((__m256i*) & result.data[i * p + j], sum);
		}
	}
}
		}
		else if constexpr (std::is_floating_point<NUMBER_TYPE>::value) {
			if constexpr (sizeof(NUMBER_TYPE) == 4) {
				for (int i = 0; i < m; i++) {
					for (int j = 0; j < p; j += 8) {
						__m256 sum = _mm256_setzero_ps();
						for (int k = 0; k < n; k++) {
							__m256 a = _mm256_set1_ps(A.data[i * n + k]);
							__m256 b = _mm256_loadu_ps(&B.data[k * p + j]);
							sum = _mm256_add_ps(sum, _mm256_mul_ps(a, b));
						}
						_mm256_storeu_ps(&result.data[i * p + j], sum);
					}
				}
			}
			else if constexpr (sizeof(NUMBER_TYPE) == 8) {
				for (int i = 0; i < m; i++) {
					for (int j = 0; j < p; j += 4) {
						__m256d sum = _mm256_setzero_pd();
						for (int k = 0; k < n; k++) {
							__m256d a = _mm256_set1_pd(A.data[i * n + k]);
							__m256d b = _mm256_loadu_pd(&B.data[k * p + j]);
							sum = _mm256_add_pd(sum, _mm256_mul_pd(a, b));
						}
						_mm256_storeu_pd(&result.data[i * p + j], sum);
					}
				}
			}
		}
#else
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < p; j++) {
				NUMBER_TYPE sum = 0;
				for (int k = 0; k < n; k++) {
					sum += A.data[i * n + k] * B.data[k * p + j];
				}
				result.data[i * p + j] = sum;
			}
		}
#endif
		if (d1 > 2 || d2 > 2) {
			vector<int> shape(d1 + d2 - 2);
			for (int i = 0; i < d1 - 2; i++) {
				shape[i] = A.shape[i];
			}
			for (int i = 0; i < d2 - 2; i++) {
				shape[static_cast<std::vector<int, std::allocator<int>>::size_type>(i) + d1 - 2] = B.shape[i];
			}
			shape[static_cast<std::vector<int, std::allocator<int>>::size_type>(d1) + d2 - 4] = m;
			shape[static_cast<std::vector<int, std::allocator<int>>::size_type>(d1) + d2 - 3] = p;
			result.resize(shape);
		}
		result.squeeze();
		return result;
	}

	Tensor<NUMBER_TYPE> tensor_product(const Tensor<NUMBER_TYPE>& other) {
		// Créer un vecteur pour stocker la nouvelle dimension du résultat
		std::vector<int> new_shape;
		// Concaténer les dimensions des deux tenseurs
		new_shape.insert(new_shape.end(), this->shape.begin(), this->shape.end());
		new_shape.insert(new_shape.end(), other.shape.begin(), other.shape.end());
		// Calculer la nouvelle taille du résultat
		int new_size = 1;
		for (int x : new_shape) {
			new_size *= x;
		}
		// Créer un vecteur pour stocker les nouvelles données du résultat
		std::vector<NUMBER_TYPE> new_data(new_size);
		// Effectuer le produit tensoriel
		for (int i = 0; i < this->length; i++) {
			for (int j = 0; j < other.length; j++) {
				// Calculer l'indice du résultat en fonction des indices des deux tenseurs
				int index = i * other.length + j;
				// Multiplier les éléments correspondants des deux tenseurs
				new_data[index] = this->data[i] * other.data[j];
			}
		}
		// Créer un nouveau tenseur avec les nouvelles données et la nouvelle dimension
		Tensor<NUMBER_TYPE> result(new_data, new_shape);
		// Retourner le résultat
		return result;
	}

	auto fft() {
		assert(!(std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value));
		int n = std::pow(2, ceil(log2(this->length)));
		Tensor<complex<NUMBER_TYPE>> result({ n });
		std::copy(this->data.begin(), this->data.end(), result.data.begin());
		FFT<NUMBER_TYPE> FFTInstance(n);
		std::vector<complex<NUMBER_TYPE>> output = FFTInstance.transform(result.data);
		std::copy(output.begin(), output.end(), result.data.begin());
		return result;
	}

	auto ifft() {
		assert((std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value));
		using COMPLEX_TYPE = NUMBER_TYPE::value_type;
		int n = std::pow(2, ceil(log2(this->length)));
		Tensor<COMPLEX_TYPE> result({ n });
		result.show();
		std::vector<NUMBER_TYPE> Data(n);
		std::copy(this->data.begin(), this->data.end(), Data.begin());
		FFT<COMPLEX_TYPE> FFTInstance(n);
		std::vector<NUMBER_TYPE> output = FFTInstance.inverse_transform(Data);
		std::transform(output.begin(), output.end(), result.data.begin(), [](NUMBER_TYPE c) { return c.real(); });
		return result;
	};

	Tensor<NUMBER_TYPE> fwht() {
		vector<int> original_shape = this->shape;
		// On aplatit le tenseur
		this->flatten();
		// On vérifie si sa longueur est une puissance de 2
		if ((this->length & (this->length - 1)) != 0) {
			// Si ce n'est pas le cas, on calcule la prochaine puissance de 2 supérieure
			int nextPowerOfTwo = std::pow(2, ceil(log2(this->length)));
			// On calcule le nombre de zéros à ajouter
			int padding = nextPowerOfTwo - this->length;
			// On redimensionne le tenseur en ajoutant des zéros à la fin
			this->pad({ 0 }, { padding });
		}
		// On copie les données du tenseur dans un vecteur
		vector<NUMBER_TYPE> data = this->data;
		// On calcule la WHT du vecteur
		::fwht(data, this->length);
		this->resize(original_shape);
		// On renvoie un nouveau tenseur avec les données transformées
		return Tensor<NUMBER_TYPE>(data, { static_cast<int>(data.size()) });
	}

	Tensor<NUMBER_TYPE> ifwht() {

		vector<int> original_shape = this->shape;
		// On aplatit le tenseur
		this->flatten();
		// On vérifie si sa longueur est une puissance de 2
		if ((this->length & (this->length - 1)) != 0) {
			// Si ce n'est pas le cas, on calcule la prochaine puissance de 2 supérieure
			int nextPowerOfTwo = std::pow(2, ceil(log2(this->length)));
			// On calcule le nombre de zéros à ajouter
			int padding = nextPowerOfTwo - this->length;
			// On redimensionne le tenseur en ajoutant des zéros à la fin
			this->pad({ 0 }, { padding });
		}
		// On copie les données du tenseur dans un vecteur
		vector<NUMBER_TYPE> data = this->data;
		// On calcule la WHT inverse du vecteur
		::ifwht(data, this->length);
		this->resize(original_shape);
		// On renvoie un nouveau tenseur avec les données transformées
		return Tensor<NUMBER_TYPE>(data, this->shape);
	}

	Tensor<NUMBER_TYPE> convolve(const int& output_dim, const int& filter_num, Tensor<NUMBER_TYPE>& kernel, vector<int> strides = {}, vector<int> dilatations = {}, string padding = "valid") {
		// Vérifier que les paramètres sont valides
		if (output_dim < 1 || output_dim > this->dimension) throw runtime_error("Invalid output dimension");
		if (filter_num < 1) throw runtime_error("Invalid filter number");
		if (kernel.dimension != output_dim + 1) throw runtime_error("Invalid kernel dimension");
		if (strides.size() != 0 && strides.size() != output_dim) throw runtime_error("Invalid strides size");
		if (dilatations.size() != 0 && dilatations.size() != output_dim) throw runtime_error("Invalid dilatations size");
		if (padding != "valid" && padding != "same" && padding != "full") throw runtime_error("Invalid padding mode");

		// Initialiser les paramètres par défaut
		if (strides.size() == 0) strides = vector<int>(output_dim, 1);
		if (dilatations.size() == 0) dilatations = vector<int>(output_dim, 1);

		// Calculer la forme du tenseur de sortie
		vector<int> output_shape = ::squeeze(convolve_output_shape(output_dim, filter_num, this->shape, kernel.shape, strides, dilatations, padding));

		// Créer le tenseur de sortie
		Tensor<NUMBER_TYPE> output(output_shape);

		// Créer une copie du tenseur d'entrée
		Tensor<NUMBER_TYPE> input = *this;

		// Ajouter du rembourrage au tenseur d'entrée si nécessaire
		vector<int> padding_size(output_dim);
		for (int i = 0; i < output_dim; i++) {
			if (padding == "valid") {
				padding_size[i] = 0;
			}
			else if (padding == "same") {
				padding_size[i] = (kernel.shape[i + 1] - 1) * dilatations[i] - (this->shape[i] - 1) % strides[i];
			}
			else if (padding == "full") {
				padding_size[i] = (kernel.shape[i + 1] - 1) * dilatations[i];
			}
		}
		input.pad(vector<int>(output_dim, 0), padding_size, vector<bool>(output_dim, false));

		// Parcourir les filtres du noyau
		for (int f = 0; f < filter_num; f++) {
			// Extraire le filtre courant du noyau
			Tensor<NUMBER_TYPE> filter = kernel.getMult({ f });

			// Parcourir les positions possibles du filtre sur le tenseur d'entrée
			vector<int> position(output_dim, 0);
			bool done = false;
			while (!done) {
				// Calculer l'index du tenseur de sortie correspondant à la position du filtre
				vector<int> output_index(output_dim + 1);
				output_index[0] = f;
				for (int i = 0; i < output_dim; i++) {
					output_index[i + 1] = position[i];
				}
				int output_linear = getIndex(output_index, output_shape);

				// Extraire la sous-région du tenseur d'entrée correspondant à la position du filtre
				vector<int> start(input.dimension);
				vector<int> end(input.dimension);
				for (int i = 0; i < input.dimension; i++) {
					start[i] = position[i] * strides[i];
					end[i] = start[i] + filter.shape[i] * dilatations[i];
				}
				Tensor<NUMBER_TYPE> subregion = input.slice(start, end);

				// Dilater la sous-région si nécessaire
				if (std::any_of(dilatations.begin(), dilatations.end(), [](int x) { return x > 1; })) {
					subregion.pad(vector<int>(output_dim, 0), vector<int>(output_dim, 0), vector<bool>(output_dim, true));
					for (int i = 0; i < output_dim; i++) {
						subregion.shift({ i }, { 0 }, { dilatations[i] - 1 });
					}
				}

				// Calculer le produit scalaire entre la sous-région et le filtre
				NUMBER_TYPE dot_product = (subregion * filter).sum();

				// Ajouter le produit scalaire au tenseur de sortie
				output.data[output_linear] += dot_product;

				// Passer à la position suivante
				int carry = 1;
				for (int i = output_dim - 1; i >= 0; i--) {
					position[i] += carry;
					if (position[i] >= output_shape[i + 1]) {
						position[i] = 0;
						carry = 1;
					}
					else {
						carry = 0;
						break;
					}
				}
				if (carry == 1) {
					done = true;
				}
			}
		}

		// Retourner le tenseur de sortie
		return output;
	}

	Tensor<NUMBER_TYPE> fast_convolve(Tensor<NUMBER_TYPE> kernel) {
		assert(!(std::is_same<NUMBER_TYPE, std::complex<double>>::value || std::is_same<NUMBER_TYPE, std::complex<int>>::value || std::is_same<NUMBER_TYPE, std::complex<long>>::value || std::is_same<NUMBER_TYPE, std::complex<float>>::value || std::is_same<NUMBER_TYPE, std::complex<long double>>::value));
		if (kernel.shape[0] > this->shape[0] || kernel.shape[1] > this->shape[1]) {
			throw std::invalid_argument("Le noyau doit être plus petit que le tenseur");
		}
		Tensor<std::complex<NUMBER_TYPE>> fft_tensor = this->copy().fft();
		Tensor<std::complex<NUMBER_TYPE>> fft_kernel = kernel.fft();
		fft_kernel.pad({ 0 }, { fft_tensor.length - fft_kernel.length });
		Tensor<std::complex<NUMBER_TYPE>> fft_product = fft_tensor * fft_kernel;
		Tensor<NUMBER_TYPE> result = fft_product.ifft();
		return result;
	}

	void operator =(const NUMBER_TYPE& value)
	{
		std::fill(data.begin(), data.end(), value);
	}

	void operator =(const vector<NUMBER_TYPE> new_data)
	{
		int size = 1;
		for (int i = 0; i < shape.size(); i++)
		{
			size *= shape[i];
		}
		if (new_data.size() != size) throw runtime_error("New data must have the same size as the old data.");
		this->length = static_cast<int>(new_data.size());
		this->data = new_data;
	}

	void operator =(const Tensor<NUMBER_TYPE>& other)
	{
		this->data = other.data;
		this->dimension = other.dimension;
		this->length = other.length;
		this->shape = other.shape;
	}

	void operator +=(const Tensor<NUMBER_TYPE>& other)
	{
		if (::squeeze(shape) != ::squeeze(other.shape)) throw runtime_error("Tensors must have the same shape to add them together.");
		std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::plus<NUMBER_TYPE>());
	}

	void operator -=(const Tensor<NUMBER_TYPE>& other)
	{
		if (shape != other.shape) throw runtime_error("Tensors must have the same shape to subtract them.");
		std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::minus<NUMBER_TYPE>());
	}

	void operator *=(const Tensor<NUMBER_TYPE>& other)
	{
		if (shape != other.shape) throw runtime_error("Tensors must have the same shape to multiply them.");
		std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::multiplies<NUMBER_TYPE>());
	}

	void operator /=(const Tensor<NUMBER_TYPE>& other)
	{
		if (shape != other.shape) throw runtime_error("Tensors must have the same shape to divide them.");
		std::transform(this->data.begin(), this->data.end(), other.data.begin(), this->data.begin(), std::divides<NUMBER_TYPE>());
	}

	Tensor<NUMBER_TYPE> operator +(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to add them together.");
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), t.data.begin(), result.data.begin(), std::plus<NUMBER_TYPE>());
		return result;
	}

	Tensor<NUMBER_TYPE> operator -(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to subtract them.");
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), t.data.begin(), result.data.begin(), std::minus<NUMBER_TYPE>());
		return result;
	}

	Tensor<NUMBER_TYPE> operator *(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to add them together.");
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), t.data.begin(), result.data.begin(), std::multiplies<NUMBER_TYPE>());
		return result;
	}

	Tensor<NUMBER_TYPE> operator /(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to divide them.");
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), t.data.begin(), result.data.begin(), std::divides<NUMBER_TYPE>());
		return result;
	}

	bool operator ==(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) return false;
		return std::equal(this->data.begin(), this->data.end(), t.data.begin());
	}

	bool operator <(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to divide them.");
		return std::lexicographical_compare(this->data.begin(), this->data.end(), t.data.begin(), t.data.end());
	}

	bool operator >(const Tensor<NUMBER_TYPE>& t)
	{
		if (this->shape != t.shape) throw runtime_error("Tensors must have the same shape to divide them.");
		return std::lexicographical_compare(t.data.begin(), t.data.end(), this->data.begin(), this->data.end());
	}

	Tensor<NUMBER_TYPE> operator +(NUMBER_TYPE t)
	{
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [t](NUMBER_TYPE x) { return x + t; });
		return result;
	}

	Tensor<NUMBER_TYPE> operator -(NUMBER_TYPE t)
	{
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [t](NUMBER_TYPE x) { return x - t; });
		return result;
	}

	Tensor<NUMBER_TYPE> operator *(NUMBER_TYPE t)
	{
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [t](NUMBER_TYPE x) { return x * t; });
		return result;
	}

	Tensor<NUMBER_TYPE> operator /(NUMBER_TYPE t)
	{
		Tensor<NUMBER_TYPE> result(this->data, this->shape);
		std::transform(this->data.begin(), this->data.end(), result.data.begin(), [t](NUMBER_TYPE x) { return x / t; });
		return result;
	}

	void operator +=(NUMBER_TYPE t)
	{
		std::transform(this->data.begin(), this->data.end(), this->data.begin(), [t](NUMBER_TYPE x) { return x + t; });
	}

	void operator -=(NUMBER_TYPE t)
	{
		std::transform(this->data.begin(), this->data.end(), this->data.begin(), [t](NUMBER_TYPE x) { return x - t; });
	}

	void operator *=(NUMBER_TYPE t)
	{
		std::transform(this->data.begin(), this->data.end(), this->data.begin(), [t](NUMBER_TYPE x) { return x * t; });
	}

	void operator /=(NUMBER_TYPE t)
	{
		std::transform(this->data.begin(), this->data.end(), this->data.begin(), [t](NUMBER_TYPE x) { return x / t; });
	}
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
struct activationFunction {
	string name;
	std::function<NUMBER_TYPE(NUMBER_TYPE)> run = {};
	std::function<NUMBER_TYPE(NUMBER_TYPE)> derivative = {};
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
activationFunction<NUMBER_TYPE> GetActivation(const char* name = "linear")
{
	activationFunction<NUMBER_TYPE> result;
	std::string s = name;
	for (int i = 0; i < s.size(); i++)
	{
		s[i] = tolower(s[i]);
	};
	result.name = s;
	if (s == "linear") {
		result.run = [](NUMBER_TYPE x) {
			return x;
			};
		result.derivative = [](NUMBER_TYPE x) {
			return 1;
			};
	}
	else if (s == "sigmoid") {
		result.run = [](NUMBER_TYPE x) {
			return 1.0 / (1.0 + std::exp(-x));
			};
		result.derivative = [](NUMBER_TYPE x) {
			return exp(x) / (1 + pow(exp(x), 2));
			};
	}
	else if (s == "tanh") {
		result.run = [](NUMBER_TYPE x) {
			return std::tanh(x);
			};
		result.derivative = [](NUMBER_TYPE x) {
			return 1 - std::pow(std::tanh(x), 2);
			};
	}
	else if (s == "relu") {
		result.run = [](NUMBER_TYPE x) {
			return (x > 0 ? x : NUMBER_TYPE(0));
			};
		result.derivative = [](NUMBER_TYPE x) {
			return (x > 0 ? NUMBER_TYPE(1) : NUMBER_TYPE(0));
			};
	}
	else if (s == "leakyrelu")
	{
		result.run = [](NUMBER_TYPE x) {
			return (x > x * 0.01 ? x : x * 0.01);
			};
		result.derivative = [](NUMBER_TYPE x) {
			return (x > 0 ? 1 : 0.01);
			};
	}
	else if (s == "softplus")
	{
		result.run = [](NUMBER_TYPE x) {
			return log(1 + exp(x));
			};
		result.derivative = [](NUMBER_TYPE x) {
			return 1.0 / (1.0 + std::exp(-x));
			};
	}
	else if (s == "gaussian")
	{
		result.run = [](NUMBER_TYPE x) {
			return exp(-(pow(x, 2)));
			};
		result.derivative = [](NUMBER_TYPE x) {
			return exp(-(pow(x, 2))) * -2 * x;
			};
	}
	else if (s == "elu") {
		result.run = [](NUMBER_TYPE x) {
			int alpha = 1;
			return (x > 0 ? x : alpha * (exp(x) - 1));
			};
		result.derivative = [](NUMBER_TYPE x) {
			int alpha = 1;
			return (x > 0 ? 1 : alpha * exp(x));
			};

	}
	else {
		throw runtime_error("The activation searched is not an activation Function, you must choose between : Linear, Sigmoid, TanH, ReLU, LeakyReLU, SoftPlus, Gaussian, Elu");
	}
	return result;
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
struct lossFunction {
	string name;
	std::function<NUMBER_TYPE(Tensor<NUMBER_TYPE>, Tensor<NUMBER_TYPE>)> run;
	std::function<Tensor<NUMBER_TYPE>(Tensor<NUMBER_TYPE>, Tensor<NUMBER_TYPE>)> derivative;
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
lossFunction<NUMBER_TYPE> GetLoss(const char* name = "mse")
{
	lossFunction<NUMBER_TYPE> result;
	std::string s = name;
	for (int i = 0; i < s.size(); i++)
	{
		s[i] = tolower(s[i]);
	};
	result.name = s;
	if (s == "mse")
	{
		result.run = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			double sum = 0;
			for (int i = 0; i < n; i++) {
				sum += pow(y.data[i] - y_pred.data[i], 2);
			}


			return sum / n;
			};
		result.derivative = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			Tensor<NUMBER_TYPE> grad(y.shape);
			for (int i = 0; i < n; i++) {
				grad.data[i] = -2.0 / n * (y.data[i] - y_pred.data[i]);


			}
			return grad;
			};
	}
	else if (s == "mae")
	{
		result.run = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			double sum = 0;
			for (int i = 0; i < n; i++) {
				sum += abs(y.data[i] - y_pred.data[i]);
			}


			return sum / n;
			};
		result.derivative = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			Tensor<NUMBER_TYPE> grad(y.shape);
			for (int i = 0; i < n; i++) {
				grad.data[i] = -1.0 / n * (y.data[i] > y_pred.data[i] ? 1 : -1);


			}
			return grad;
			};
	}
	else if (s == "bce")
	{
		result.run = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			double sum = 0;
			for (int i = 0; i < n; i++) {
				sum += y.data[i] * log(y_pred.data[i]) + (1 - y.data[i]) * log(1 - y_pred.data[i]);
			}


			return -sum / n;
			};
		result.derivative = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			Tensor<NUMBER_TYPE> grad(y.shape);
			for (int i = 0; i < n; i++) {
				grad.data[i] = -1.0 / n * (y.data[i] / y_pred.data[i] - (1 - y.data[i]) / (1 - y_pred.data[i]));


			}
			return grad;
			};
	}
	else if (s == "cce")
	{
		result.run = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			double sum = 0;
			for (int i = 0; i < n; i++) {
				sum += y.data[i] * log(y_pred.data[i]);
			}


			return -sum / n;
			};
		result.derivative = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			Tensor<NUMBER_TYPE> grad(y.shape);
			int n = y.length;
			for (int i = 0; i < n; i++) {
				grad.data[i] = -1.0 / n * y.data[i] / y_pred.data[i];


			}
			return grad;
			};
	}
	else if (s == "ji")
	{
		result.run = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			int interSize = 0;
			int unionSize = 0;
			for (int i = 0; i < n; i++) {
				if (y.data[i] && y_pred.data[i]) interSize++;
				if (y.data[i] || y_pred.data[i]) unionSize++;
			}
			return double(interSize) / unionSize;
			};
		result.derivative = [](Tensor<NUMBER_TYPE> y, Tensor<NUMBER_TYPE> y_pred) {
			int n = y.length;
			int interSize = 0;
			int unionSize = 0;
			for (int i = 0; i < n; i++) {
				if (y.data[i] && y_pred.data[i]) interSize++;
				if (y.data[i] || y_pred.data[i]) unionSize++;
			}
			Tensor<NUMBER_TYPE> grad(y.shape);
			for (int i = 0; i < n; i++) {
				grad.data[i] = 1.0 / pow(unionSize, 2) * (interSize + unionSize * (y.data[i] ? 1 : -1));


			}
			return grad;
			};
	}

	else {
		throw runtime_error("The activation searched is not an loss Function, you must choose between : MSE, MAE, BCE, CCE, JI");
	}
	return result;
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
struct optimizationAlgorithm {
	string name;
	int batch_size = 16;
	double learningRate = 0.1;
	vector<double> options = {};
	vector<Tensor<NUMBER_TYPE>> m_w;
	vector<Tensor<NUMBER_TYPE>> m_b;
	vector<Tensor<NUMBER_TYPE>> v_w;
	vector<Tensor<NUMBER_TYPE>> v_b;
	std::function<void(vector<Tensor<NUMBER_TYPE>>&, vector<Tensor<NUMBER_TYPE>>&, vector<Tensor<NUMBER_TYPE>>&, vector<Tensor<NUMBER_TYPE>>&, int)> run;
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
optimizationAlgorithm<NUMBER_TYPE> GetOptimization(const char* name = "GD", vector<double> options = { 0.1 }, int batch_size = 0)
{
	optimizationAlgorithm<NUMBER_TYPE> result;
	std::string s = name;
	for (int i = 0; i < s.size(); i++)
	{
		s[i] = tolower(s[i]);
	};
	result.name = s;
	result.batch_size = batch_size;
	if (options[0] != NULL) result.learningRate = options[0];
	if (batch_size < 0) throw runtime_error("Batch Size parameter cannot be negative");
	if (s == "gd")
	{
		if (options.size() != 1) throw runtime_error("The GD optimization Algorithm only have 1 parameter : Learning Rate (0 to 1)");
		if (batch_size != 0) throw runtime_error("The GD optimization Algorithm have no Batch Size parameter else it's the SGD Algorithm");
		result.run = [lr = &result.learningRate](vector<Tensor<NUMBER_TYPE>>& weights, vector<Tensor<NUMBER_TYPE>>& bias, vector<Tensor<NUMBER_TYPE>>& gWeights, vector<Tensor<NUMBER_TYPE>>& gBias, int epoch) {
			for (int i = 0; i < weights.size(); i++)
			{
				weights[i] -= gWeights[i] * (*lr);
				bias[i] -= gBias[i] * (*lr);
			}
			};
	}
	else if (s == "sgd")
	{
		if (options.size() != 1) throw runtime_error("The SGD optimization Algorithm only have 2 parameter : Learning Rate (float, 0 to 1), Batch Size (int)");
		if (batch_size == 0) throw runtime_error("The SGD optimization Algorithm have a Batch Size parameter else it's the GD Algorithm");
		result.run = [lr = &result.learningRate](vector<Tensor<NUMBER_TYPE>>& weights, vector<Tensor<NUMBER_TYPE>>& bias, vector<Tensor<NUMBER_TYPE>>& gWeights, vector<Tensor<NUMBER_TYPE>>& gBias, int epoch) {
			for (int i = 0; i < weights.size(); i++)
			{
				weights[i] -= gWeights[i] * (*lr);
				bias[i] -= gBias[i] * (*lr);
			}
			};
	}
	else if (s == "gdm")
	{
		if (options.size() != 2) throw runtime_error("The SGD optimization Algorithm only have 2 parameter : Learning Rate (float, 0 to 1), Gamma (float, 0 to 1)");
		result.run = [lr = &result.learningRate, &gamma = options[1], v_w = &result.v_w, v_b = &result.v_b](vector<Tensor<NUMBER_TYPE>>& weights, vector<Tensor<NUMBER_TYPE>>& bias, vector<Tensor<NUMBER_TYPE>>& gWeights, vector<Tensor<NUMBER_TYPE>>& gBias, int epoch) {
			int n = static_cast<int>(weights.size());
			v_w->resize(n);
			v_b->resize(bias.size());
			for (int i = 0; i < n; i++)
			{
				(*v_w)[i] = Tensor<NUMBER_TYPE>(weights[i].shape);
				(*v_b)[i] = Tensor<NUMBER_TYPE>(bias[i].shape);
			}
			for (int i = 0; i < n; i++)
			{
				(*v_w)[i] = (*v_w)[i] * gamma + gWeights[i] * (1 - gamma);
				(*v_b)[i] = (*v_b)[i] * gamma + gBias[i] * (1 - gamma);
			}
			for (int i = 0; i < n; i++)
			{
				weights[i] -= (*v_w)[i] * (*lr);
				bias[i] -= (*v_b)[i] * (*lr);
			}
			};
	}
	else if (s == "adam")
	{
		if (options.size() != 4) throw runtime_error("The Adam optimization Algorithm only have 4 parameters : Learning Rate (float, 0 to 1), Moment strides (float, 0 to 1), RMSProp strides (float, 0 to 1), Numerical Stability (little float, 0 to 1e-05)");

		result.run = [lr = &result.learningRate, &epsilon = options[3], &beta1 = options[1], &beta2 = options[2], v_w = &result.v_w, v_b = &result.v_b, m_w = &result.m_w, m_b = &result.m_b](vector<Tensor<NUMBER_TYPE>>& weights, vector<Tensor<NUMBER_TYPE>>& bias, vector<Tensor<NUMBER_TYPE>>& gWeights, vector<Tensor<NUMBER_TYPE>>& gBias, int epoch) {
			int n = static_cast<int>(weights.size());
			m_w->resize(n); m_b->resize(n);
			v_w->resize(n); v_b->resize(n);
			for (int i = 0; i < n; i++)
			{
				(*m_w)[i] = Tensor<NUMBER_TYPE>(weights[i].shape);
				(*m_b)[i] = Tensor<NUMBER_TYPE>(bias[i].shape);
				(*v_w)[i] = Tensor<NUMBER_TYPE>(weights[i].shape);
				(*v_b)[i] = Tensor<NUMBER_TYPE>(bias[i].shape);
			}
			for (int i = 0; i < n; i++)
			{
				(*m_w)[i] = (*m_w)[i] * beta1 + gWeights[i] * (1 - beta1);
				(*m_b)[i] = (*m_b)[i] * beta1 + gBias[i] * (1 - beta1);
				(*v_w)[i] = (*v_w)[i] * beta2 + gWeights[i] * gWeights[i] * (1 - beta2);
				(*v_b)[i] = (*v_b)[i] * beta2 + gBias[i] * gBias[i] * (1 - beta2);

				weights[i] = weights[i] - (*m_w)[i] * (*lr) / ((*v_w)[i].sqrt() + epsilon);
				bias[i] = bias[i] - (*m_b)[i] * (*lr) / ((*v_b)[i].sqrt() + epsilon);
			};
			};
	}
	else throw runtime_error("The optimization searched is not an optimization Algorithm, you must choose between : GD, SGD, GDM, Adam");
	return result;
}

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
class DenseLayer {
private:
	activationFunction<NUMBER_TYPE> activation;
public:
	bool builded = false;
	int output_size;
	int input_size;
	Tensor<NUMBER_TYPE> lastInputs;
	Tensor<NUMBER_TYPE> lastOutputs;
	Tensor<NUMBER_TYPE> weights;
	Tensor<NUMBER_TYPE> bias;

	DenseLayer(int neurons = 1, activationFunction<NUMBER_TYPE> act = {
		"sigmoid",
		[](NUMBER_TYPE x) { return 1.0 / (1.0 + std::exp(-x)); },
		[](NUMBER_TYPE x) { return exp(x) / (1 + pow(exp(x), 2)); }
		})
	{
		this->input_size = 1;
		this->output_size = neurons;
		this->activation = act;
	}

	DenseLayer(int input_size = 1, int output_size = 1, activationFunction<NUMBER_TYPE> act = {
		"sigmoid",
		[](NUMBER_TYPE x) { return 1.0 / (1.0 + std::exp(-x)); },
		[](NUMBER_TYPE x) { return exp(x) / (1 + pow(exp(x), 2)); }
		})
	{
		this->output_size = output_size;
		this->activation = act;
		this->input_size = input_size;
		this->builded = true;
	}

	Tensor<NUMBER_TYPE> forward(Tensor<NUMBER_TYPE> input) {

		lastInputs = input;

		if (input.length != input_size) {
			std::cout << "Erreur : la taille du vecteur d’entrée ne correspond pas à la taille des entrées de la couche." << endl; return {};
		}
		Tensor<NUMBER_TYPE> z = Tensor<NUMBER_TYPE>(lastInputs.data, { 1,lastInputs.length });
		lastOutputs = z.matmul(weights) + bias;
		Tensor<NUMBER_TYPE> outputs = lastOutputs.apply(activation.run);

		outputs.resize({ output_size });
		return outputs;
	}

	Tensor<NUMBER_TYPE> backward(Tensor<NUMBER_TYPE> dL_dy, Tensor<NUMBER_TYPE>& dL_dw, Tensor<NUMBER_TYPE>& dL_db) {
		dL_dy.flatten();
		dL_db.resize({ output_size });
		dL_dw.resize({ output_size, input_size });
		Tensor<NUMBER_TYPE> dL_dx({ input_size });

		Tensor<NUMBER_TYPE> dy_dz = lastOutputs.apply(activation.derivative);
		dL_db = dL_dy * dy_dz;
		dL_dw = dL_db.transpose() * lastInputs;
		dL_dx = weights.transpose() * dL_db;

		dL_dx.resize(lastInputs.shape);
		return dL_dx;
	}

	Tensor<NUMBER_TYPE> build(Tensor<NUMBER_TYPE> input_size)
	{
		this->input_size = input_size.length;
		this->weights.resize({ input_size.length, this->output_size });
		this->bias.resize({ this->output_size });
		this->weights.randomize();
		this->bias.randomize();
		this->builded = true;
		return Tensor<NUMBER_TYPE>({ this->output_size });
	}
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
class ConvolutionnalLayer {

private:
	activationFunction<NUMBER_TYPE> activation;
public:
	vector<int> input_size;
	int output_dim;
	int filter_num;
	vector<int> kernel_size;
	vector<int> strides;
	vector<int> dilatation;
	string padding;
	bool builded = false;
	Tensor<NUMBER_TYPE> lastInputs;
	Tensor<NUMBER_TYPE> lastOutputs;
	Tensor<NUMBER_TYPE> filters;
	Tensor<NUMBER_TYPE> bias;

	ConvolutionnalLayer(vector<int> _input_size, int _output_dim, int _filter_num, vector<int> _kernel_size, vector<int> _strides, vector<int> _dilatation, string _padding = "valid", activationFunction<NUMBER_TYPE> act = {
		"relu",
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? x : 0); }),
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? 1 : 0); })
		})
	{
		this->input_size = _input_size;
		this->output_dim = _output_dim;
		this->filter_num = _filter_num;
		this->padding = _padding;
		this->activation = act;
		int maxi = max(_kernel_size.size(), max(_strides.size(), max(_input_size.size(), _dilatation.size())));
		this->kernel_size = _kernel_size;
		if (this->input_size.size() - this->output_dim > 0) this->kernel_size.insert(this->kernel_size.end(), _input_size.end() - _input_size.size() - _output_dim, _input_size.end());
		if (this->kernel_size.empty() || this->kernel_size.size() < maxi) this->kernel_size.resize(maxi, 1);
		this->strides = _strides;
		if (this->strides.empty() || this->strides.size() < maxi) this->strides.resize(maxi, 1);
		this->dilatation = _dilatation;
		if (this->dilatation.empty() || this->dilatation.size() < maxi) this->dilatation.resize(maxi, 1);
		vector<int> size;
		size.push_back(filter_num);
		std::copy(_kernel_size.begin(), _kernel_size.end(), std::back_inserter(size));
		this->filters = Tensor<NUMBER_TYPE>(size);
		this->bias = Tensor<NUMBER_TYPE>(squeeze<int>(convolution_size(this->filter_num, this->input_size, this->kernel_size, this->strides, this->dilatation, this->padding)));
		this->filters.randomize();
		this->bias.randomize();
		this->builded = true;
	}

	ConvolutionnalLayer(int _output_dim, int _filter_num, vector<int> _kernel_size, vector<int> _strides, vector<int> _dilatation, string _padding = "valid", activationFunction<NUMBER_TYPE> act = {
		"relu",
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? x : 0); }),
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? 1 : 0); })
		})
	{
		this->input_size = { 1 };
		this->output_dim = _output_dim;
		this->filter_num = _filter_num;
		this->padding = _padding;
		this->activation = act;
		this->kernel_size = { filter_num };
		this->kernel_size.insert(this->kernel_size.end(), _kernel_size.begin(), _kernel_size.end());
		this->strides = _strides;
		this->dilatation = _dilatation;
	}

	Tensor<NUMBER_TYPE> forward(Tensor<NUMBER_TYPE> input) {
		lastInputs = input;
		lastOutputs = lastInputs.convolve(this->output_dim, this->filter_num, this->filters, this->strides, this->dilatation, this->padding);
		lastOutputs += this->bias;
		return lastOutputs.apply(this->activation.run);
	}

	Tensor<NUMBER_TYPE> backward(Tensor<NUMBER_TYPE> dL_dy, Tensor<NUMBER_TYPE>& dL_dw, Tensor<NUMBER_TYPE>& dL_db) {
		return Tensor<NUMBER_TYPE>();
	}

	Tensor<NUMBER_TYPE> build(Tensor<NUMBER_TYPE> input)
	{
		this->input_size = input.shape;
		/*
		int maxi = max(this->kernel_size.size() - 1, max(this->strides.size(), max(this->input_size.size(), this->dilatation.size())));
		if (this->kernel_size.size()-1 != this->input_size.size()) {
			this->kernel_size.resize(this->input_size.size()+1);
		}
		for (int i = this->output_dim; i < this->input_size.size(); i++) {
			if (this->kernel_size[i + 1] != this->input_size[i]) {
				this->kernel_size[i + 1] = this->input_size[i];
			}
		}
		if (this->strides.empty() || this->strides.size() < maxi) this->strides.resize(maxi, 1);
		if (this->dilatation.empty() || this->dilatation.size() < maxi) this->dilatation.resize(maxi, 1);
		*/
		vector<int> conv = convolve_output_shape(this->output_dim, this->filter_num, this->input_size, this->kernel_size, this->strides, this->dilatation, this->padding);
		this->bias = Tensor<NUMBER_TYPE>(conv);
		this->filters = Tensor<NUMBER_TYPE>(kernel_size);
		this->filters.randomize();
		this->bias.randomize();
		this->builded = true;
		return Tensor<NUMBER_TYPE>(bias.shape);
	}
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
class FastConvolutionnalLayer {

private:
	activationFunction<NUMBER_TYPE> activation;
public:
	int input_size; // Taille des entrées carrées
	int filter_num; // Nombres de filtres
	bool builded = false;
	Tensor<NUMBER_TYPE> lastInputs; // Dernières entrées
	Tensor<NUMBER_TYPE> lastOutputs; // Dernières sorties
	Tensor<NUMBER_TYPE> weights; // Matrice des poids du filtre
	Tensor<NUMBER_TYPE> bias; // Vecteur du biais

	FastConvolutionnalLayer(int input_size, int filter_num, activationFunction<NUMBER_TYPE> act = {
			"relu",
			std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? x : 0); }),
			std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? 1 : 0); })
		}) {
		this->input_size = input_size;
		this->filter_num = filter_num;
		this->activation = act;
		this->weights = Tensor<NUMBER_TYPE>({ this->filter_num, input_size, input_size });
		this->bias = Tensor<NUMBER_TYPE>({ this->filter_num });
		this->weights.randomize(); // Initialiser les poids aléatoirement
		this->bias.randomize(); // Initialiser les biais aléatoirement
		this->builded = true;
	}

	FastConvolutionnalLayer(int filter_num, activationFunction<NUMBER_TYPE> act = {
		"relu",
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? x : 0); }),
		std::function<NUMBER_TYPE(NUMBER_TYPE)>([](NUMBER_TYPE x) { return (x > 0 ? 1 : 0); })
		})
	{
		this->input_size = 1;
		this->filter_num = filter_num;
		this->activation = act;
		this->bias = Tensor<NUMBER_TYPE>({ this->filter_num });
		this->bias.randomize();
	}

	Tensor<NUMBER_TYPE> forward(Tensor<NUMBER_TYPE> input) {
		if ((input.shape[0] != this->input_size || input.shape[1] != this->input_size) && !this->builded) {
			std::cout << "Erreur : la taille de la matrice d'entrée ne correspond pas à la taille des entrées de la couche." << endl;
			return {};
		}

		lastInputs = input;
		lastInputs.pad({ 1, 2 }, { this->input_size - input.shape[1], this->input_size - input.shape[2] });
		lastOutputs = Tensor<NUMBER_TYPE>({ this->filter_num, this->input_size, this->input_size });

		for (int i = 0; i < this->filter_num; i++) {
			Tensor<NUMBER_TYPE> convolution = lastInputs.getMult({ i }).fast_convolve(this->weights.getMult({ i }));
			convolution = convolution + this->bias.get({ i });
			convolution = convolution.apply(this->activation.run);
			lastOutputs.setMult({ i }, convolution);
		}
		return lastOutputs;
	}

	Tensor<NUMBER_TYPE> backward(Tensor<NUMBER_TYPE> dL_dy, Tensor<NUMBER_TYPE>& dL_dw, Tensor<NUMBER_TYPE>& dL_db) {
		Tensor<NUMBER_TYPE> dL_dx(lastInputs.shape);
		return dL_dx;
	}

	Tensor<NUMBER_TYPE> build(Tensor<NUMBER_TYPE> input)
	{
		this->builded = true;
		int n = std::pow(2, ceil(log2(input.length)));
		this->input_size = n;
		this->weights = Tensor<NUMBER_TYPE>({ this->filter_num, n, n });
		this->weights.randomize();
		return Tensor<NUMBER_TYPE>({ this->filter_num, n, n });
	}
};

template <typename NUMBER_TYPE = double>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
class DropoutLayer {

public:
	double dropout_rate;
	Tensor<int> mask;
	DropoutLayer(double rate = 0.1)
	{
		this->dropout_rate = rate;
	}

	Tensor<NUMBER_TYPE> forward(Tensor<NUMBER_TYPE> input) {
		Tensor<NUMBER_TYPE> outputs = input;
		for (int i = 0; i < input.length; i++)
		{
			int n = randomUniform<int>(0, 2);
			this->mask.data[i] = n;
			if (n == 0)
			{
				outputs.data[i] = NUMBER_TYPE(0);
			}
		}
		return outputs;
	}

	Tensor<NUMBER_TYPE> backward(Tensor<NUMBER_TYPE> dL_dy, Tensor<NUMBER_TYPE>& dL_dw, Tensor<NUMBER_TYPE>& dL_db) {

		Tensor<NUMBER_TYPE> dL_dx(this->mask.shape);

		dL_dx = dL_dy * this->mask;

		return dL_dx;
	}

	Tensor<NUMBER_TYPE> build(Tensor<NUMBER_TYPE> input_size)
	{
		this->mask.resize({ input_size.length });
		return Tensor<NUMBER_TYPE>({ input_size.length });
	}
};

template <typename NUMBER_TYPE = double, typename INPUT = double, typename OUTPUT = double, typename INPUT_TRAINING = INPUT, typename OUTPUT_TRAINING = OUTPUT>
	requires std::is_floating_point_v<NUMBER_TYPE> || std::is_integral_v<NUMBER_TYPE>
class Sequential
{
private:
	std::vector<std::variant<DenseLayer<NUMBER_TYPE>, FastConvolutionnalLayer<NUMBER_TYPE>, ConvolutionnalLayer<NUMBER_TYPE>, DropoutLayer<NUMBER_TYPE>>> layers;
	bool trainable = true;
	bool builded = false;
public:
	int layerNum = 0;
	vector<int> input_shape = { 1 };
	vector<int> output_shape = { 1 };
	Tensor<NUMBER_TYPE>(*convertInput)(INPUT&) = [](INPUT& x) {
		return x;
		};
	OUTPUT(*convertOutput)(Tensor<NUMBER_TYPE>&) = [](Tensor<NUMBER_TYPE>& x) {
		return x;
		};
	Tensor<NUMBER_TYPE>(*convertOutputToData)(OUTPUT_TRAINING&) = [](OUTPUT_TRAINING& x) {
		return x;
		};
	Tensor<NUMBER_TYPE>(*convertInputToData)(INPUT_TRAINING&) = [](INPUT_TRAINING& x) {
		return x;
		};

	Sequential(vector<int> input_shape = { 1 }, vector<int> output_shape = { 1 })
	{
		this->input_shape = input_shape;
		this->output_shape = output_shape;
	};

	void build()
	{
		this->builded = true;
		Tensor<NUMBER_TYPE> lastShape = this->input_shape;
		for (int i = 0; i < layerNum; i++)
		{
			cout << "building " << i << endl;
			Tensor<NUMBER_TYPE> result;
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = dense->build(lastShape);
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = fast->build(lastShape);
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = conv->build(lastShape);
			}
			else if (DropoutLayer<NUMBER_TYPE>* drop = std::get_if<DropoutLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = drop->build(lastShape);
			}
			lastShape = result;
		}
	}

	void add(std::variant<DenseLayer<NUMBER_TYPE>, FastConvolutionnalLayer<NUMBER_TYPE>, ConvolutionnalLayer<NUMBER_TYPE>, DropoutLayer<NUMBER_TYPE>> layer)
	{
		this->layerNum++;
		this->layers.push_back(layer);
		this->builded = false;
	}

	vector<Tensor<NUMBER_TYPE>> getWeights()
	{
		if (!this->builded) this->build();
		vector<Tensor<NUMBER_TYPE>> weights;
		weights.resize(layerNum);
		for (int i = 0; i < layerNum; i++)
		{
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				weights[i] = dense->weights;
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				weights[i] = fast->weights;
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				weights[i] = conv->weights;
			}
			else
			{
				weights[i] = Tensor<NUMBER_TYPE>({ 0 });
			}
		}
		return weights;
	}

	void setWeights(vector<Tensor<NUMBER_TYPE>> weights)
	{
		if (!this->builded) this->build();
		weights.resize(layerNum);
		for (int i = 0; i < layerNum; i++)
		{
			if (weights[i].shape[0] == 0) continue;
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				dense->weights = weights[i];
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				fast->weights = weights[i];
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				conv->weights = weights[i];
			}
		}
	}

	vector<Tensor<NUMBER_TYPE>> getBias()
	{
		if (!this->builded) this->build();
		vector<Tensor<NUMBER_TYPE>> bias;
		bias.resize(layerNum);
		for (int i = 0; i < layerNum; i++)
		{
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				bias[i] = dense->bias;
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				bias[i] = fast->bias;
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				bias[i] = conv->bias;
			}
			else
			{
				bias[i] = Tensor<NUMBER_TYPE>({ 0 });
			}
		}
		return bias;
	}

	void setBias(vector<Tensor<NUMBER_TYPE>> bias)
	{
		if (!this->builded) this->build();
		bias.resize(layerNum);
		for (int i = 0; i < layerNum; i++)
		{
			if (bias[i].shape[0] == 0) continue;
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				dense->bias = bias[i];
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				fast->bias = bias[i];
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				conv->bias = bias[i];
			}
		}
	}

	Tensor<NUMBER_TYPE> forward(Tensor<NUMBER_TYPE> input)
	{
		if (!this->builded)
		{
			this->build();
		}
		Tensor<NUMBER_TYPE> lastInputs = input;
		for (int i = 0; i < layerNum; i++)
		{
			cout << "Launch Layer " << i << endl;
			Tensor<NUMBER_TYPE> result;
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = dense->forward(lastInputs);
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = fast->forward(lastInputs);
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = conv->forward(lastInputs);
			}
			else if (DropoutLayer<NUMBER_TYPE>* drop = std::get_if<DropoutLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = drop->forward(lastInputs);
			}
			lastInputs = result;
		}
		return lastInputs;
	}

	OUTPUT run(INPUT input)
	{
		Tensor<NUMBER_TYPE> convertedInput = convertInput(input);
		Tensor<NUMBER_TYPE> result = this->forward(convertedInput);
		return convertOutput(result);
	}

	void backward(Tensor<NUMBER_TYPE> dL_dy, vector<Tensor<NUMBER_TYPE>>& dL_dw, vector<Tensor<NUMBER_TYPE>>& dL_db)
	{
		if (!this->builded) this->build();
		Tensor<NUMBER_TYPE> lastGradientsInputs = dL_dy;
		for (int i = layerNum - 1; i >= 0; i--)
		{
			Tensor<NUMBER_TYPE> result;
			Tensor<NUMBER_TYPE> dW({ 0 });
			Tensor<NUMBER_TYPE> dB({ 0 });
			if (DenseLayer<NUMBER_TYPE>* dense = std::get_if<DenseLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = dense->backward(lastGradientsInputs, dW, dB);
			}
			else if (FastConvolutionnalLayer<NUMBER_TYPE>* fast = std::get_if<FastConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = fast->backward(lastGradientsInputs, dW, dB);
			}
			else if (ConvolutionnalLayer<NUMBER_TYPE>* conv = std::get_if<ConvolutionnalLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = conv->backward(lastGradientsInputs, dW, dB);
			}
			else if (DropoutLayer<NUMBER_TYPE>* drop = std::get_if<DropoutLayer<NUMBER_TYPE>>(&layers[i]))
			{
				result = drop->backward(lastGradientsInputs, dW, dB);
			}
			dL_dw[i] = dW;
			dL_db[i] = dB;
			lastGradientsInputs = result;
		}
	}

	string train(vector<pair<INPUT_TRAINING, OUTPUT_TRAINING>>& trainingSet, lossFunction<NUMBER_TYPE> lossFunction, optimizationAlgorithm<NUMBER_TYPE> optiAlgorithm, vector<double>* allLoss = nullptr, vector<double>* allValidLoss = nullptr)
	{
		if (!this->builded) this->build();
		if (!this->trainable)
		{
			cerr << "This model is actually training or can't be train now, restart later please" << endl;
			return "Error";
		}

		this->trainable = false;

		string state;

		// float lambda = static_cast<float>(0.01); // regularization coefficient

		float baseLoss = INFINITY;
		float bestLoss = INFINITY;

		float baseValidLoss = INFINITY;
		float lastValidLoss = INFINITY;
		float bestValidLoss = INFINITY;

		vector<Tensor<NUMBER_TYPE>> dL_dw(layerNum);
		vector<Tensor<NUMBER_TYPE>> dL_db(layerNum);

		vector<Tensor<NUMBER_TYPE>> weights = this->getWeights();
		vector<Tensor<NUMBER_TYPE>> bias = this->getBias();

		vector<Tensor<NUMBER_TYPE>> bestWeights = weights;
		vector<Tensor<NUMBER_TYPE>> bestBias = bias;

		for (int i = 0; i < layerNum; i++)
		{
			dL_dw[i].resize(weights[i].shape);
			dL_db[i].resize(bias[i].shape);
		}

		int failcount = 0;
		bool epochContinue = true;
		int trainingSetSize = static_cast<int>(trainingSet.size());
		int maxBatchSize = int(trainingSetSize / 2);
		vector<int> lossTrainingSet = range(0, maxBatchSize);
		vector<int> validTrainingSet = range(maxBatchSize, maxBatchSize * 2);

		int batchSize = optiAlgorithm.batch_size;
		if (batchSize <= 0) batchSize = 1;
		if (batchSize > maxBatchSize) batchSize = maxBatchSize;
		int batchNum = static_cast<int>(lossTrainingSet.size() / batchSize);
		for (int epoch = 0; epochContinue; epoch++)
		{
			// Shuffle training set
			shuffle(lossTrainingSet);

			float loss = 0.0;
			float validLoss = 0.0;

			for (int batch = 0; batch < batchNum; batch++)
			{
				Tensor<NUMBER_TYPE> lossDerivative(this->output_shape);
				for (int i = 0; i < layerNum; i++)
				{
					dL_dw[i].reset();
					dL_db[i].reset();
				}
				for (int i = 0; i < batchSize; i++)
				{
					if (isnan(validLoss) || isinf(validLoss) && epoch > 50)
					{
						epochContinue = false;
						validLoss = static_cast<double>(INFINITY);
						failcount = static_cast<int>(INFINITY);
						break;
					}
					int Index = getIndex({ batch, i }, { batchNum, batchSize });
					// Add regularization term to loss
					pair<INPUT_TRAINING, OUTPUT_TRAINING> validObject = trainingSet[validTrainingSet[Index]];
					float reg_loss = 0.0;
					Tensor<NUMBER_TYPE> validOutput = convertOutputToData(validObject.second);
					Tensor<NUMBER_TYPE> validResult = this->forward(convertInputToData(validObject.first));
					validLoss += float(lossFunction.run(validOutput, validResult)) + reg_loss;

					pair<INPUT_TRAINING, OUTPUT_TRAINING> object = trainingSet[lossTrainingSet[Index]];
					Tensor<NUMBER_TYPE> output = convertOutputToData(object.second);
					Tensor<NUMBER_TYPE> result = this->forward(convertInputToData(object.first));
					lossDerivative += lossFunction.derivative(output, result);
					loss += float(lossFunction.run(output, result)) + reg_loss;
				}
				loss /= batchSize;
				lossDerivative /= batchSize;
				if (epoch != 0)
				{
					this->backward(lossDerivative, dL_dw, dL_db);

					optiAlgorithm.run(weights, bias, dL_dw, dL_db, epoch);

					this->setWeights(weights);
					this->setBias(bias);
				}
				validLoss /= batchSize;
			};
			loss /= batchNum;
			validLoss /= batchNum;
			if (epoch == 0)
			{
				baseLoss = loss;
				baseValidLoss = validLoss;
				bestLoss = loss;
				bestValidLoss = validLoss;
			}
			else {
				if (validLoss > baseValidLoss && epoch > 50)
				{
					epochContinue = false;
				}
				if (bestValidLoss < validLoss || std::to_string(validLoss) == std::to_string(lastValidLoss))
				{
					failcount++;
					// optiAlgorithm.learningRate /= 1.0025;
				}
				else
				{
					bestBias = bias;
					bestWeights = weights;
					bestLoss = loss;
					bestValidLoss = validLoss;
					failcount = 0;
					// optiAlgorithm.learningRate *= 1.0025;
				}
				if (failcount >= 50)
				{
					this->setBias(bestBias);
					this->setWeights(bestWeights);
					epochContinue = false;
				}
			}
			if (allLoss != nullptr) allLoss->push_back(loss);
			if (allValidLoss != nullptr) allValidLoss->push_back(validLoss);
			cout << "epoch " << epoch << " ; loss = " << loss << " ; validLoss = " << validLoss << " ; fail = " << failcount << endl;
			lastValidLoss = validLoss;
		}
		this->trainable = true;
		state = (bestLoss >= baseLoss) ? ((bestValidLoss >= baseValidLoss) ? "Fail Learning" : "Underfitting") : ((bestValidLoss >= baseValidLoss) ? "Overfitting" : "Good Learning");
		return state;
	};
};

int main2()
{
	srand(0);
	int exampleNumber = 5000;
	float limit = 1;
	float start = -1;
	vector<int> input_size = { 64,32,21 }; // 64,32,21
	vector<int> output_size = { 7 }; // 7
	Sequential<float, Tensor<float>, Tensor<float>> ann(input_size, output_size);

	ann.convertInput = [](Tensor<float>& x) {
		return x;
		};
	ann.convertInputToData = [](Tensor<float>& x) {
		return x;
		};
	ann.convertOutput = [](Tensor<float>& x) {
		return x;
		};
	ann.convertOutputToData = [](Tensor<float>& x) {
		return x;
		};
	/*
	ann.add(DenseLayer(1, GetActivation("linear")));

	std::cout << ann.forward(Tensor({ 0.5 }, { 1 })).get({ 0 }) << " result 0.5" << endl;
	std::cout << ann.forward(Tensor({ 0.2 }, { 1 })).get({ 0 }) << " result 0.2" << endl;

	vector<pair<Tensor<double>, Tensor<double>>> trainingSet(exampleNumber);
	for (int i = 0; i < exampleNumber; i++) {
		double num = (start + double(i) / exampleNumber * (limit - start));
		trainingSet[i] = make_pair(Tensor({ num }, { 1 }), Tensor({ num * 2 * 1.0 }, { 1 }));
	}
	auto startTime = std::chrono::steady_clock::now();

	string state = ann.train(trainingSet, GetLoss("mse"), GetOptimization("SGD", { 1e-3 }, 4));
	std::cout << "L'état du modèle est " + state << endl;

	auto endTime = std::chrono::steady_clock::now();
	auto diff = endTime - startTime;
	std::cout << "Le temps d'exécution est " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms" << std::endl;

	std::cout << ann.forward(Tensor({ 0.5 }, { 1 })).get({ 0 }) << " result 0.5" << endl;
	std::cout << ann.forward(Tensor({ 0.2 }, { 1 })).get({ 0 }) << " result 0.2" << endl;
	*/
	const char* ELU_char = "elu";
	vector<int> kernel_size = { 5, 5 };
	vector<int> kernel_size2 = { 3, 3 };
	vector<int> strides = { 2, 2 };
	vector<int> strides2 = { 1, 1 };
	vector<int> dilatation = { 2, 2 };
	ann.add(ConvolutionnalLayer<float>(2, 24, kernel_size, strides, dilatation, "valid", GetActivation<float>(ELU_char)));
	ann.add(ConvolutionnalLayer<float>(2, 36, kernel_size, strides, dilatation, "valid", GetActivation<float>(ELU_char)));
	ann.add(ConvolutionnalLayer<float>(2, 48, kernel_size, strides, dilatation, "valid", GetActivation<float>(ELU_char)));
	ann.add(ConvolutionnalLayer<float>(2, 64, kernel_size2, strides2, dilatation, "valid", GetActivation<float>(ELU_char)));
	ann.add(ConvolutionnalLayer<float>(2, 64, kernel_size2, strides2, dilatation, "valid", GetActivation<float>(ELU_char)));
	ann.add(DropoutLayer<float>(0.2));
	ann.add(DenseLayer<float>(100, GetActivation<float>("elu")));
	ann.add(DropoutLayer<float>(0.2));
	ann.add(DenseLayer<float>(50, GetActivation<float>("elu")));
	ann.add(DropoutLayer<float>(0.2));
	ann.add(DenseLayer<float>(10, GetActivation<float>("elu")));
	ann.add(DropoutLayer<float>(0.2));
	ann.add(DenseLayer<float>(output_size[0], GetActivation<float>("linear")));
	Tensor<float> a(input_size);
	a.randomize();
	ann.build();
	cout << "go" << endl;
	auto startTime = std::chrono::steady_clock::now();
	Tensor<float> b = ann.forward(a);
	cout << "fini" << endl;
	auto endTime = std::chrono::steady_clock::now();
	auto diff = endTime - startTime;
	std::cout << "Le temps d'exécution est " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms" << std::endl;
	/*
	ann.add(FastConvolutionnalLayer<float>(output_size[0], GetActivation<float>("linear")));
	ann.add(FastConvolutionnalLayer<float>(output_size[0], GetActivation<float>("linear")));
	ann.add(FastConvolutionnalLayer<float>(output_size[0], GetActivation<float>("linear")));
	ann.add(FastConvolutionnalLayer<float>(output_size[0], GetActivation<float>("linear")));
	Tensor<float> t = Tensor<float>(input_size);
	t.randomize();
	ann.build();
	auto startTime = std::chrono::steady_clock::now();
	Tensor<float> r = ann.forward(t);
	auto endTime = std::chrono::steady_clock::now();
	auto diff = endTime - startTime;
	std::cout << "Le temps d'exécution est " << std::chrono::duration_cast<std::chrono::milliseconds>(diff).count() << " ms" << std::endl;

	*/
	return 0;
}

int main()
{
	Tensor<int> a(range(1, 12 + 1), { 3,4 });
	Tensor<int> b({ 1,0,0,1 }, { 1,2,2 });
	Tensor<int> c = a.convolve(2, 1, b, { 1,1 }, { 1,1 }, "valid");
	c.show();
}