/**************************************************************/
/********* SerializableDoubleVector Implementation *********/
/**************************************************************/

#include <vector>       // std::vector
#include <cstddef>      // size_t
#include <cstdint>      // uint64_t

// Dragon headers - these cover Serializable, dragonFLISendHandleDescr_t,
// dragonFLIRecvHandleDescr_t, timespec_t, and DragonError
#include <dragon/serializable.hpp>

#include "cpp_serializers.hpp"

namespace custom {

SerializableDoubleVector::SerializableDoubleVector(std::vector<double> x): mVal(x) {}
SerializableDoubleVector::SerializableDoubleVector(size_t size): mVal(std::vector<double>(size, 0.0)) {}

void SerializableDoubleVector::serialize(dragonFLISendHandleDescr_t* sendh, uint64_t arg, const bool buffer, const timespec_t* timeout) const {
    dragonError_t err;
    size_t num_items = mVal.size();

    // write the size of the array - needed for efficient deserialization without copies
    err = dragon_fli_send_bytes(sendh, sizeof(size_t), (uint8_t*)&num_items, arg, buffer, timeout);
    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not write vector size to ddict.");

    err = dragon_fli_send_bytes(sendh, sizeof(double)*num_items, (uint8_t*)mVal.data(), arg, buffer, timeout);
    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not write vector to ddict.");
}

SerializableDoubleVector SerializableDoubleVector::deserialize(dragonFLIRecvHandleDescr_t* recvh, uint64_t* arg, const timespec_t* timeout) {
    dragonError_t err = DRAGON_SUCCESS;
    size_t received_size = 0;
    size_t expected_size = 0;
    size_t num_items = 0;

    err = dragon_fli_recv_bytes_into(recvh, sizeof(size_t), &received_size, (uint8_t*)&num_items, arg, timeout);
    if (err == DRAGON_TIMEOUT)
        throw dragon::TimeoutError(err, "Operation timeout.");

    if (err == DRAGON_EOT)
        throw dragon::EmptyError(err, "EOT of stream");

    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not read item count for vector.");

    if (received_size != sizeof(size_t))
        throw dragon::DragonError(DRAGON_INVALID_ARGUMENT, "The size of num_items was not correct.");

    SerializableDoubleVector rv(num_items);

    expected_size = sizeof(double) * num_items;

    err = dragon_fli_recv_bytes_into(recvh, expected_size, &received_size, (uint8_t*)rv.mVal.data(), arg, timeout);
    if (err == DRAGON_TIMEOUT)
        throw dragon::TimeoutError(err, "Operation timeout.");

    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not read element of vector.");

    if (received_size != expected_size)
        throw dragon::DragonError(DRAGON_INVALID_ARGUMENT, "The received data did not match the expected size of the vector.");

    return rv; // Relies on RVO for efficiently returning the vector.
}

const std::vector<double>& SerializableDoubleVector::getVal() const {return mVal;}

/**************************************************************/
/********* SerializableDouble2DVector Implementation *********/
/**************************************************************/

SerializableDouble2DVector::SerializableDouble2DVector(std::vector<std::vector<double>> x): mVal(x) {}

void SerializableDouble2DVector::serialize(dragonFLISendHandleDescr_t* sendh, uint64_t arg, const bool buffer, const timespec_t* timeout) const {
    dragonError_t err;
    size_t nrows = mVal.size();

    // write the number of rows in the vector
    err = dragon_fli_send_bytes(sendh, sizeof(size_t), (uint8_t*)&nrows, arg, buffer, timeout);
    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not write bytes to ddict.");

    for (auto& row: mVal) {
        SerializableDoubleVector sVec(row);
        sVec.serialize(sendh, arg, buffer, timeout);
    }
}

SerializableDouble2DVector SerializableDouble2DVector::deserialize(dragonFLIRecvHandleDescr_t* recvh, uint64_t* arg, const timespec_t* timeout) {
    dragonError_t err = DRAGON_SUCCESS;
    size_t actual_size;
    std::vector<std::vector<double>> val;

    size_t nrows = 0;

    err = dragon_fli_recv_bytes_into(recvh, sizeof(size_t), &actual_size, (uint8_t*)&nrows, arg, timeout);
    if (err == DRAGON_TIMEOUT)
        throw dragon::TimeoutError(err, "Operation timeout.");

    if (err == DRAGON_EOT)
        throw dragon::EmptyError(err, "EOT of stream");

    if (err != DRAGON_SUCCESS)
        throw dragon::DragonError(err, "Could not read row count for vector.");

    if (actual_size != sizeof(size_t))
        throw dragon::DragonError(DRAGON_INVALID_ARGUMENT, "The size of nrows was not correct.");

    for (size_t i=0; i<nrows; i++) {
        SerializableDoubleVector tmp_vec = SerializableDoubleVector::deserialize(recvh, arg, timeout);
        val.push_back(tmp_vec.getVal());
    }

    return SerializableDouble2DVector(val); // Relies on RVO for efficiently returning the vector.
}

const std::vector<std::vector<double>>& SerializableDouble2DVector::getVal() const {return mVal;}

}