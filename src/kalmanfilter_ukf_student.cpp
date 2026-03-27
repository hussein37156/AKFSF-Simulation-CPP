// ------------------------------------------------------------------------------- //
// Advanced Kalman Filtering and Sensor Fusion Course - Unscented Kalman Filter
//
// ####### STUDENT FILE #######
//
// Usage:
// -Rename this file to "kalmanfilter.cpp" if you want to use this code.

#include "kalmanfilter.h"
#include "utils.h"

// -------------------------------------------------- //
// YOU CAN USE AND MODIFY THESE CONSTANTS HERE
constexpr double ACCEL_STD = 1.0;
constexpr double GYRO_STD = 0.01 / 180.0 * M_PI;
constexpr double INIT_VEL_STD = 10.0;
constexpr double INIT_PSI_STD = 45.0 / 180.0 * M_PI;
constexpr double GPS_POS_STD = 3.0;
constexpr double LIDAR_RANGE_STD = 3.0;
constexpr double LIDAR_THETA_STD = 0.02;
// -------------------------------------------------- //

// ----------------------------------------------------------------------- //
// USEFUL HELPER FUNCTIONS
VectorXd normaliseState(VectorXd state)
{
    state(2) = wrapAngle(state(2));
    return state;
}
VectorXd normaliseLidarMeasurement(VectorXd meas)
{
    meas(1) = wrapAngle(meas(1));
    return meas;
}
std::vector<VectorXd> generateSigmaPoints(VectorXd state, MatrixXd cov)
{
    std::vector<VectorXd> sigmaPoints;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    int n_x = state.size();
    double k = 3.0 - n_x;
    MatrixXd sqrtCov = cov.llt().matrixL();
    sigmaPoints.push_back(state);
    for (size_t i = 0; i < n_x; i++)
    {
        sigmaPoints.push_back(state + sqrt(n_x + k) * sqrtCov.col(i));
        sigmaPoints.push_back(state - sqrt(n_x + k) * sqrtCov.col(i));
    }

    // ----------------------------------------------------------------------- //

    return sigmaPoints;
}

std::vector<double> generateSigmaWeights(unsigned int numStates)
{
    std::vector<double> weights;

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    double k = 3.0 - numStates;
    double w0 = k / (numStates + k);
    double w1 = 0.5 / (numStates + k);
    weights.push_back(w0);
    for (size_t i = 0; i < 2 * numStates; i++)
    {
        weights.push_back(w1);
    }

    // ----------------------------------------------------------------------- //

    return weights;
}

VectorXd lidarMeasurementModel(VectorXd aug_state, double beaconX, double beaconY)
{
    VectorXd z_hat = VectorXd::Zero(2);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    double x_diff = beaconX - aug_state(0);
    double y_diff = beaconY - aug_state(1);
    z_hat(0) = sqrt(x_diff * x_diff + y_diff * y_diff) + aug_state(4);
    z_hat(1) = atan2(y_diff , x_diff) - aug_state(2) + aug_state(5);

    // ----------------------------------------------------------------------- //

    return z_hat;
}

VectorXd vehicleProcessModel(VectorXd aug_state, double psi_dot, double dt)
{
    VectorXd new_state = VectorXd::Zero(4);

    // ----------------------------------------------------------------------- //
    // ENTER YOUR CODE HERE
    new_state << aug_state(0) + dt * (aug_state(3) * cos(aug_state(2))),
        aug_state(1) + dt * (aug_state(3) * sin(aug_state(2))),
        aug_state(2) + dt * (psi_dot + aug_state(4)),
        aug_state(3) + dt * aug_state(5);
    // ----------------------------------------------------------------------- //

    return new_state;
}
// ----------------------------------------------------------------------- //

void KalmanFilter::handleLidarMeasurement(LidarMeasurement meas, const BeaconMap &map)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Update Step for the Lidar Measurements in the
        // section below.
        // HINT: Use the normaliseState() and normaliseLidarMeasurement() functions
        // to always keep angle values within correct range.
        // HINT: Do not normalise during sigma point calculation!
        // HINT: You can use the constants: LIDAR_RANGE_STD, LIDAR_THETA_STD
        // HINT: The mapped-matched beacon position can be accessed by the variables
        // map_beacon.x and map_beacon.y
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        BeaconData map_beacon = map.getBeaconWithId(meas.id); // Match Beacon with built in Data Association Id
        if (meas.id != -1 && map_beacon.id != -1)             // Check that we have a valid beacon match
        {
            // Generate Measurement Vector
            VectorXd z = Vector2d::Zero();
            z << meas.range, meas.theta;

            // Generate Measurement Model Noise Covariance Matrix
            MatrixXd R = Matrix2d::Zero();
            R(0, 0) = LIDAR_RANGE_STD * LIDAR_RANGE_STD;
            R(1, 1) = LIDAR_THETA_STD * LIDAR_THETA_STD;

            // Augment the State Vector with Noise States
            int n_x = state.size();
            int n_v = 2;
            int n_z = 2;
            int n_aug = n_x + n_v;
            VectorXd state_aug = VectorXd::Zero(n_aug);
            MatrixXd cov_aug = MatrixXd::Zero(n_aug, n_aug);
            state_aug.head(n_x) = state;
            cov_aug.topLeftCorner(n_x, n_x) = cov;
            cov_aug.bottomRightCorner(n_v, n_v) = R;


            std::vector<VectorXd> sigma_points = generateSigmaPoints(state_aug, cov_aug);
            std::vector<double> sigma_weights = generateSigmaWeights(n_aug);

            std::vector<VectorXd> sigma_points_predict;
            for (const auto &sigma_point : sigma_points)
            {
                sigma_points_predict.push_back(lidarMeasurementModel(sigma_point, map_beacon.x, map_beacon.y));
            }

            VectorXd z_mean = VectorXd::Zero(n_z);
            for (unsigned int i = 0; i < sigma_points_predict.size(); ++i)
            {
                z_mean += sigma_weights[i] * sigma_points_predict[i];
            }

            MatrixXd innov_cov = MatrixXd::Zero(n_z, n_z);
            for (unsigned int i = 0; i < sigma_points_predict.size(); ++i)
            {
                VectorXd diff = normaliseLidarMeasurement(sigma_points_predict[i] - z_mean);
                innov_cov += sigma_weights[i] * diff * diff.transpose();
            }

            MatrixXd cross_cov = MatrixXd::Zero(n_x, n_z);
            for (unsigned int i = 0; i < 2 * n_x + 1; ++i)
            {
                VectorXd x_diff = normaliseState(sigma_points[i].head(n_x) - state);
                VectorXd z_diff = normaliseLidarMeasurement(sigma_points_predict[i] - z_mean);
                cross_cov += sigma_weights[i] * x_diff * z_diff.transpose();
            }

            MatrixXd K = cross_cov * innov_cov.inverse();
            VectorXd innov_error = normaliseLidarMeasurement(z - z_mean);
            state = state + K * innov_error;
            cov = cov - K * innov_cov * K.transpose();
        }
        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::predictionStep(GyroMeasurement gyro, double dt)
{
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        // Implement The Kalman Filter Prediction Step for the system in the
        // section below.
        // HINT: Assume the state vector has the form [PX, PY, PSI, V].
        // HINT: Use the Gyroscope measurement as an input into the prediction step.
        // HINT: You can use the constants: ACCEL_STD, GYRO_STD
        // HINT: Use the normaliseState() function to always keep angle values within correct range.
        // HINT: Do NOT normalise during sigma point calculation!
        // ----------------------------------------------------------------------- //
        // ENTER YOUR CODE HERE

        int n_x = state.size();
        int n_w = 2;
        int n_aug = n_w + n_x;

        MatrixXd Q = MatrixXd::Zero(n_w, n_w);
        Q(0, 0) = GYRO_STD * GYRO_STD;
        Q(1, 1) = ACCEL_STD * ACCEL_STD;

        VectorXd state_aug = VectorXd::Zero(n_aug);
        MatrixXd cov_aug = MatrixXd::Zero(n_aug, n_aug);

        state_aug.head(n_x) = state;
        cov_aug.topLeftCorner(n_x, n_x) = cov;
        cov_aug.bottomRightCorner(n_w, n_w) = Q;

        std::vector<VectorXd> sigma_points = generateSigmaPoints(state_aug, cov_aug);
        std::vector<double> sigma_weights = generateSigmaWeights(n_aug);

        std::vector<VectorXd> sigma_points_predict;
        for (auto sigma_point : sigma_points)
        {
            sigma_points_predict.push_back(vehicleProcessModel(sigma_point, gyro.psi_dot, dt));
        }

        state = VectorXd::Zero(n_x);
        for (size_t i = 0; i < sigma_points_predict.size(); i++)
        {
            state += sigma_weights[i] * sigma_points_predict[i];
        }
        state = normaliseState(state);

        cov = MatrixXd::Zero(n_x, n_x);
        for (size_t i = 0; i < sigma_points_predict.size(); i++)
        {
            VectorXd diff = normaliseState(sigma_points_predict[i] - state);
            cov += sigma_weights[i] * diff * diff.transpose();
        }

        // ----------------------------------------------------------------------- //

        setState(state);
        setCovariance(cov);
    }
}

void KalmanFilter::handleGPSMeasurement(GPSMeasurement meas)
{
    // All this code is the same as the LKF as the measurement model is linear
    // so the UKF update state would just produce the same result.
    if (isInitialised())
    {
        VectorXd state = getState();
        MatrixXd cov = getCovariance();

        VectorXd z = Vector2d::Zero();
        MatrixXd H = MatrixXd(2, 4);
        MatrixXd R = Matrix2d::Zero();

        z << meas.x, meas.y;
        H << 1, 0, 0, 0, 0, 1, 0, 0;
        R(0, 0) = GPS_POS_STD * GPS_POS_STD;
        R(1, 1) = GPS_POS_STD * GPS_POS_STD;

        VectorXd z_hat = H * state;
        VectorXd y = z - z_hat;
        MatrixXd S = H * cov * H.transpose() + R;
        MatrixXd K = cov * H.transpose() * S.inverse();

        state = state + K * y;
        cov = (MatrixXd::Identity(4, 4) - K * H) * cov;

        setState(state);
        setCovariance(cov);
    }
    else
    {
        // You may modify this initialisation routine if you can think of a more
        // robust and accuracy way of initialising the filter.
        // ----------------------------------------------------------------------- //
        // YOU ARE FREE TO MODIFY THE FOLLOWING CODE HERE

        VectorXd state = Vector4d::Zero();
        MatrixXd cov = Matrix4d::Zero();

        state(0) = meas.x;
        state(1) = meas.y;
        cov(0, 0) = GPS_POS_STD * GPS_POS_STD;
        cov(1, 1) = GPS_POS_STD * GPS_POS_STD;
        cov(2, 2) = INIT_PSI_STD * INIT_PSI_STD;
        cov(3, 3) = INIT_VEL_STD * INIT_VEL_STD;

        setState(state);
        setCovariance(cov);

        // ----------------------------------------------------------------------- //
    }
}

void KalmanFilter::handleLidarMeasurements(const std::vector<LidarMeasurement> &dataset, const BeaconMap &map)
{
    // Assume No Correlation between the Measurements and Update Sequentially
    for (const auto &meas : dataset)
    {
        handleLidarMeasurement(meas, map);
    }
}

Matrix2d KalmanFilter::getVehicleStatePositionCovariance()
{
    Matrix2d pos_cov = Matrix2d::Zero();
    MatrixXd cov = getCovariance();
    if (isInitialised() && cov.size() != 0)
    {
        pos_cov << cov(0, 0), cov(0, 1), cov(1, 0), cov(1, 1);
    }
    return pos_cov;
}

VehicleState KalmanFilter::getVehicleState()
{
    if (isInitialised())
    {
        VectorXd state = getState(); // STATE VECTOR [X,Y,PSI,V,...]
        return VehicleState(state[0], state[1], state[2], state[3]);
    }
    return VehicleState();
}

void KalmanFilter::predictionStep(double dt) {}
