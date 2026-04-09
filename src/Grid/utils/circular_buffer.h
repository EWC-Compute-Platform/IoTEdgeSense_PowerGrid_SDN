/**
 * @file circular_buffer.h
 * @brief Circular buffer implementation for data storage
 * 
 * This file provides a template-based circular buffer implementation
 * for efficient storage of fixed-size data with FIFO behavior.
 */

#ifndef CIRCULAR_BUFFER_H
#define CIRCULAR_BUFFER_H

#include <cstdint>
#include <vector>
#include <mutex>
#include <stdexcept>
#include <cstring>

namespace Utils {

/**
 * @brief Template-based circular buffer class
 * 
 * @tparam T Type of data to store
 */
template <typename T>
class CircularBuffer {
public:
    /**
     * @brief Constructor
     * 
     * @param capacity Buffer capacity
     * @param threadSafe Whether to use thread synchronization
     */
    CircularBuffer(size_t capacity, bool threadSafe = true)
        : mCapacity(capacity),
          mSize(0),
          mHead(0),
          mTail(0),
          mThreadSafe(threadSafe),
          mBuffer(capacity) {
        if (capacity == 0) {
            throw std::invalid_argument("Circular buffer capacity must be greater than 0");
        }
    }
    
    /**
     * @brief Destructor
     */
    ~CircularBuffer() {
        clear();
    }
    
    /**
     * @brief Push data to the buffer
     * 
     * @param item Item to push
     * @param overwrite Whether to overwrite when full
     * @return true if push successful, false if buffer full and overwrite disabled
     */
    bool push(const T& item, bool overwrite = true) {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        if (isFull()) {
            if (!overwrite) {
                return false;
            }
            
            // Overwrite oldest item
            advanceTail();
        }
        
        mBuffer[mHead] = item;
        advanceHead();
        
        return true;
    }
    
    /**
     * @brief Pop data from the buffer
     * 
     * @param item Reference to store popped item
     * @return true if pop successful, false if buffer empty
     */
    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        if (isEmpty()) {
            return false;
        }
        
        item = mBuffer[mTail];
        advanceTail();
        
        return true;
    }
    
    /**
     * @brief Peek at the next item without removing it
     * 
     * @param item Reference to store peeked item
     * @return true if peek successful, false if buffer empty
     */
    bool peek(T& item) const {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        if (isEmpty()) {
            return false;
        }
        
        item = mBuffer[mTail];
        return true;
    }
    
    /**
     * @brief Peek at an item at specific position
     * 
     * @param index Index relative to the tail
     * @param item Reference to store peeked item
     * @return true if peek successful, false if index out of bounds
     */
    bool peekAt(size_t index, T& item) const {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        if (index >= mSize) {
            return false;
        }
        
        size_t pos = (mTail + index) % mCapacity;
        item = mBuffer[pos];
        return true;
    }
    
    /**
     * @brief Clear the buffer
     */
    void clear() {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        mHead = 0;
        mTail = 0;
        mSize = 0;
    }
    
    /**
     * @brief Check if buffer is empty
     * 
     * @return true if empty, false otherwise
     */
    bool isEmpty() const {
        return mSize == 0;
    }
    
    /**
     * @brief Check if buffer is full
     * 
     * @return true if full, false otherwise
     */
    bool isFull() const {
        return mSize == mCapacity;
    }
    
    /**
     * @brief Get current buffer size
     * 
     * @return Number of items in buffer
     */
    size_t size() const {
        return mSize;
    }
    
    /**
     * @brief Get buffer capacity
     * 
     * @return Maximum number of items buffer can hold
     */
    size_t capacity() const {
        return mCapacity;
    }
    
    /**
     * @brief Get all items as a vector
     * 
     * @return Vector containing all items in order
     */
    std::vector<T> getAll() const {
        std::unique_lock<std::mutex> lock(mMutex, std::defer_lock);
        if (mThreadSafe) lock.lock();
        
        std::vector<T> result;
        result.reserve(mSize);
        
        for (size_t i = 0; i < mSize; ++i) {
            size_t pos = (mTail + i) % mCapacity;
            result.push_back(mBuffer[pos]);
        }
        
        return result;
    }
    
    /**
     * @brief Get free space in buffer
     * 
     * @return Number of items that can be added before full
     */
    size_t freeSpace() const {
        return mCapacity - mSize;
    }

private:
    size_t mCapacity;           ///< Maximum buffer capacity
    size_t mSize;               ///< Current number of items
    size_t mHead;               ///< Write position
    size_t mTail;               ///< Read position
    bool mThreadSafe;           ///< Whether to use thread safety
    std::vector<T> mBuffer;     ///< Actual buffer data
    mutable std::mutex mMutex;  ///< Mutex for thread safety
    
    /**
     * @brief Advance head pointer
     */
    void advanceHead() {
        mHead = (mHead + 1) % mCapacity;
        if (mSize < mCapacity) {
            ++mSize;
        }
    }
    
    /**
     * @brief Advance tail pointer
     */
    void advanceTail() {
        mTail = (mTail + 1) % mCapacity;
        if (mSize > 0) {
            --mSize;
        }
    }
};

/**
 * @brief Specialized circular buffer for byte data
 */
class ByteCircularBuffer {
public:
    /**
     * @brief Constructor
     * 
     * @param capacity Buffer capacity in bytes
     * @param threadSafe Whether to use thread synchronization
     */
    ByteCircularBuffer(size_t capacity, bool threadSafe = true);
    
    /**
     * @brief Destructor
     */
    ~ByteCircularBuffer();
    
    /**
     * @brief Write data to the buffer
     * 
     * @param data Data to write
     * @param size Size of data in bytes
     * @param overwrite Whether to overwrite when full
     * @return Number of bytes written
     */
    size_t write(const uint8_t* data, size_t size, bool overwrite = true);
    
    /**
     * @brief Read data from the buffer
     * 
     * @param data Buffer to read into
     * @param size Maximum number of bytes to read
     * @return Number of bytes read
     */
    size_t read(uint8_t* data, size_t size);
    
    /**
     * @brief Peek at data without removing
     * 
     * @param data Buffer to peek into
     * @param size Maximum number of bytes to peek
     * @return Number of bytes peeked
     */
    size_t peek(uint8_t* data, size_t size) const;
    
    /**
     * @brief Skip bytes in the buffer
     * 
     * @param size Number of bytes to skip
     * @return Number of bytes skipped
     */
    size_t skip(size_t size);
    
    /**
     * @brief Clear the buffer
     */
    void clear();
    
    /**
     * @brief Check if buffer is empty
     * 
     * @return true if empty, false otherwise
     */
    bool isEmpty() const;
    
    /**
     * @brief Check if buffer is full
     * 
     * @return true if full, false otherwise
     */
    bool isFull() const;
    
    /**
     * @brief Get current buffer size
     * 
     * @return Number of bytes in buffer
     */
    size_t size() const;
    
    /**
     * @brief Get buffer capacity
     * 
     * @return Maximum number of bytes buffer can hold
     */
    size_t capacity() const;
    
    /**
     * @brief Get free space in buffer
     * 
     * @return Number of bytes that can be added before full
     */
    size_t freeSpace() const;
    
    /**
     * @brief Find a byte pattern in the buffer
     * 
     * @param pattern Pattern to search for
     * @param patternSize Size of pattern in bytes
     * @return Position of pattern relative to tail, or -1 if not found
     */
    ssize_t find(const uint8_t* pattern, size_t patternSize) const;
    
    /**
     * @brief Read until a specific byte pattern
     * 
     * @param data Buffer to read into
     * @param maxSize Maximum size of buffer
     * @param pattern Pattern to search for
     * @param patternSize Size of pattern in bytes
     * @param includePattern Whether to include pattern in read data
     * @return Number of bytes read, or -1 if pattern not found
     */
    ssize_t readUntil(uint8_t* data, size_t maxSize, const uint8_t* pattern, 
                      size_t patternSize, bool includePattern = false);

private:
    size_t mCapacity;           ///< Maximum buffer capacity
    size_t mSize;               ///< Current number of bytes
    size_t mHead;               ///< Write position
    size_t mTail;               ///< Read position
    bool mThreadSafe;           ///< Whether to use thread safety
    uint8_t* mBuffer;           ///< Actual buffer data
    mutable std::mutex mMutex;  ///< Mutex for thread safety
    
    /**
     * @brief Advance head pointer
     * 
     * @param amount Amount to advance
     */
    void advanceHead(size_t amount);
    
    /**
     * @brief Advance tail pointer
     * 
     * @param amount Amount to advance
     */
    void advanceTail(size_t amount);
    
    /**
     * @brief Get continuous write size
     * 
     * @return Size of continuous memory block for writing
     */
    size_t getContinuousWriteSize() const;
    
    /**
     * @brief Get continuous read size
     * 
     * @return Size of continuous memory block for reading
     */
    size_t getContinuousReadSize() const;
    
    /**
     * @brief Write to buffer without bounds checking
     * 
     * @param data Data to write
     * @param size Size of data
     */
    void writeInternal(const uint8_t* data, size_t size);
    
    /**
     * @brief Read from buffer without bounds checking
     * 
     * @param data Buffer to read into
     * @param size Size to read
     */
    void readInternal(uint8_t* data, size_t size);
};

} // namespace Utils

#endif // CIRCULAR_BUFFER_H

