from enum import Enum


class MLKeys(Enum):
    """
    This enum class contains the most commonly used tags in the mlde package
    """
    dir_x = 0
    dir_y = 1
    x_file = 2
    y_file = 3
    y_labels = 4
    x_labels = 5
    CFCParameters = 6
    MFCParameters = 7
    ModelName = 8
    AdditionalInfo = 9
    training = 10
    validation = 11
    train_and_validate = 12
    feature_importance = 13
    full_features = 14
    importance_sorted = 15
    chunks = 16
    model = 17
    features = 18
    timepoints = 19
    window = 20
    offset = 21
    id = 22
    features_correlation = 23
    tsfresh_selected = 24
    scikit_importance = 25
    fc_selected = 26
    target = 27
    jobs = 28
    print_score = 29
    fc_parameters = 30
    n_jobs = 31
    tsfresh_features = ['abs_energy', 'absolute_maximum', 'absolute_sum_of_changes', 'agg_autocorrelation',
                        'agg_linear_trend', 'approximate_entropy', 'ar_coefficient', 'augmented_dickey_fuller',
                        'autocorrelation', 'benford_correlation', 'binned_entropy', 'c3', 'change_quantiles',
                        'cid_ce', 'count_above', 'count_above_mean', 'count_below_mean', 'cwt_coefficients',
                        'energy_ratio_by_chunks', 'fft_aggregated', 'fft_coefficient', 'first_location_of_maximum',
                        'fourier_entropy', 'friedrich_coefficients', 'has_duplicate', 'has_duplicate_max',
                        'has_duplicate_min', 'index_mass_quantile', 'kurtosis', 'large_standard_deviation',
                        'last_location_of_maximum', 'last_location_of_minimum', 'lempel_ziv_complexity',
                        'length', 'linear trend', ' linear_trend_timewise', 'longest_strike_above_mean',
                        'longest_stike_below_mean',
                        'matrix_profile', 'max_langevin_fixed_point', 'maximum', 'mean', 'mean_abs_change',
                        'mean_change',
                        'mean_n_absolute_max', 'mean_second_derivative_central', 'median', 'minimum',
                        'number_crossing_m',
                        'number_cwt_peaks', 'number_peaks', 'partial_autocorrection',
                        'percentage_of_reoccuring_datapoints_to_all_datapoints',
                        'percentage_of_reoccurring_values_to_all_values', 'permutation_entropy', 'quantile',
                        'query_similarity_count',
                        'range_count', 'ratio_beyond_r_sigma', 'ratio_value_number_to_time_series_length',
                        'root_mean_square',
                        'sample_entropy', 'set_property', 'skewness', 'spkt_welch_density', 'standard_deviation',
                        'sum_of_reoccurring_data_points',
                        'sum_of_reoccurring_values', 'sum_values', 'symmetry_looking',
                        'time_reversal_asymmetry_statistic',
                        'value_count', 'variance', 'variance_larger_than_standard_deviation', 'variation_coefficient']




