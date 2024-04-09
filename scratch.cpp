// static double get_delta0(const std::vector<TabooPoint> &archive)
	// {
	// 	if (archive.empty())
	// 		return 1.0;

	// 	std::vector<double> radii(archive.size(), 0.0);

	// 	for (size_t i = 0; i < archive.size(); i++)
	// 	{
	// 		radii[i] = archive[i].radius;
	// 	}
	// 	const auto q1 = radii.size() / 4;
	// 	std::nth_element(radii.begin(), radii.begin() + q1, radii.end());
	// 	return radii[q1];
	// }


    // {
	// 	const double m = static_cast<double>(archive.size());
	// 	const double i = static_cast<double>(p.stats.solutions.size());
	// 	const double a_new = 0.5;
	// 	const double theta = a_new * (i / m);
	// 	const double tau = 1.0 / std::sqrt(p.settings.dim);

	// 	// std::cout << "Updating archive \n";
	// 	// std::cout << "theta: " << theta << "\n";
	// 	// std::cout << "tau: " << tau << "\npoints:\n";

	// 	for (auto &point : archive)
	// 	{
	// 		// std::cout << "delta before: " << point.delta << "\n";
	// 		// std::cout << "n_rep before: " << point.n_rep << "\n";
	// 		if (point.shares_basin(candidate_point.x, p))
	// 		{
	// 			point.n_rep++;
	// 			accept_candidate = false;

	// 			if (point.solution > candidate_point)
	// 			{
	// 				point.solution = candidate_point;
	// 			}
	// 			// std::cout << "shares basin: " << point.n_rep << "\n";
	// 			// point.delta *= std::pow(1 + point.n_rep - theta, tau);
	// 		}
	// 		// else if (point.n_rep < theta)
	// 		// {
	// 		// 	point.delta *= std::pow(1 - point.n_rep + theta, -tau);
	// 		// }

	// 		// std::cout << "delta after: " << point.delta << "\n";
	// 		// std::cout << "n_rep after: " << point.n_rep << "\n\n";

	// 		if (point.n_rep > theta)
	// 		{
	// 			point.delta = std::min(0.01, point.delta * std::pow(1 + point.n_rep - theta, tau));
	// 		}
	// 		else
	// 		{
	// 			point.delta = point.delta * std::pow(1 - point.n_rep + theta, -tau);
	// 		}
	// 	}

	// 	// std::cout << "\n";
	// 	if (accept_candidate)
	// 	{
	// 		const double delta0 = get_delta0(archive);
	// 		archive.emplace_back(candidate_point, delta0);
	// 	}
	// }
    
		double normalized(const Vector &u, const Vector &v, const parameters::Parameters &p)
		{
			if (
				p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::NONE ||
				p.settings.modules.matrix_adaptation == parameters::MatrixAdaptationType::MATRIX)
			{
				constexpr double stretch = 1.0;
				std::cout << "warning: stretch not defined\n";
				return distance::euclidian(u, v) / (p.mutation->sigma * stretch);
			}

			using namespace matrix_adaptation;
			const auto dynamic = std::dynamic_pointer_cast<CovarianceAdaptation>(p.adaptation);
			return distance::mahanolobis(u, v, dynamic->inv_C) / p.mutation->sigma;
		}