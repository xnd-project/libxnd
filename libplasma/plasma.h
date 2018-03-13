#ifdef __cplusplus
extern "C"
{
#endif
    struct ObjectID_;
    typedef struct ObjectID_ * ObjectID;
    ObjectID objectID_from_random();
    const uint8_t * objectID_data(ObjectID v);
    const char * objectID_hex(ObjectID v);

    // void * plasmaClient_connect( const char * store_socket_name,  const char * manager_socket_name, int release_delay, int num_retries);
    // int plasmaClient_create( void * v, int i );
#ifdef __cplusplus
}
#endif
