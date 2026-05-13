/**
 * @class SerializableDoubleVector
 * @brief A Serializable Vector of Doubles
 *
 * The class provides the Serializable interface for a vector of doubles.
 */

 #include <dragon/serializable.hpp>

 namespace custom {
 
 class SerializableDoubleVector : public dragon::Serializable {
     public:
     /**
      * @brief Constructor for Serializable Vector of Doubles
      *
      * This provides a wrapper class for a vector of double values that need to be serialized/deserialized in a
      * Dragon program.
      *
      * @param vec A double vector value to wrap.
      */
     SerializableDoubleVector(std::vector<double> vec);
 
     /**
      * @brief Constructor for Serializable Vector of Doubles
      *
      * Contruct an empty serializable double vector with size elements.
      *
      * @param size The number of elements for the empty vector.
      */
     SerializableDoubleVector(size_t size);
 
     /**
      * @brief See the DerivedSerializable serialize description.
      */
     virtual void serialize(dragonFLISendHandleDescr_t* sendh, uint64_t arg, const bool buffer, const timespec_t* timeout) const;
 
     /**
      * @brief See the DerivedSerializable deserialize description.
     */
     static SerializableDoubleVector deserialize(dragonFLIRecvHandleDescr_t* recvh, uint64_t* arg, const timespec_t* timeout);
 
     /**
      * @brief Get the wrapped value for the object.
      *
      * @returns The wrapped value.
      */
     const std::vector<double>& getVal() const;
 
     private:
     std::vector<double> mVal;
 };
 
 /**
  * @class SerializableDouble2DVector
  * @brief A Serializable 2D Vector of Doubles
  *
  * The class provides the Serializable interface for a 2D vector of doubles.
  */
 class SerializableDouble2DVector : public dragon::Serializable {
     public:
     /**
      * @brief Constructor for Serializable 2D Doubles
      *
      * This provides a wrapper class for a 2D vector of double values that need to be serialized/deserialized in a
      * Dragon program.
      *
      * @param vec An double vector value to wrap.
      */
     SerializableDouble2DVector(std::vector<std::vector<double>> vec);
 
     /**
      * @brief See the DerivedSerializable serialize description.
      */
     virtual void serialize(dragonFLISendHandleDescr_t* sendh, uint64_t arg, const bool buffer, const timespec_t* timeout) const;
 
     /**
      * @brief See the DerivedSerializable deserialize description.
     */
     static SerializableDouble2DVector deserialize(dragonFLIRecvHandleDescr_t* recvh, uint64_t* arg, const timespec_t* timeout);
 
     /**
      * @brief Get the wrapped value for the object.
      *
      * @returns The wrapped value.
      */
     const std::vector<std::vector<double>>& getVal() const;
 
     private:
     std::vector<std::vector<double>> mVal;
 };
 
 }