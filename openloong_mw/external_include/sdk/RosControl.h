// Generated by gencpp from file sdk/RosControl.msg
// DO NOT EDIT!


#ifndef SDK_MESSAGE_ROSCONTROL_H
#define SDK_MESSAGE_ROSCONTROL_H


#include <string>
#include <vector>
#include <memory>

#include <ros/types.h>
#include <ros/serialization.h>
#include <ros/builtin_message_traits.h>
#include <ros/message_operations.h>


namespace sdk
{
template <class ContainerAllocator>
struct RosControl_
{
  typedef RosControl_<ContainerAllocator> Type;

  RosControl_()
    : time_stamp(0)
    , arm_mode(0)
    , ee_pose_l()
    , ee_pose_r()
    , q_exp_l()
    , q_exp_r()
    , cap_set()
    , hand_q_left()
    , hand_q_right()  {
      ee_pose_l.assign(0.0);

      ee_pose_r.assign(0.0);

      q_exp_l.assign(0.0);

      q_exp_r.assign(0.0);

      cap_set.assign(0.0);

      hand_q_left.assign(0.0);

      hand_q_right.assign(0.0);
  }
  RosControl_(const ContainerAllocator& _alloc)
    : time_stamp(0)
    , arm_mode(0)
    , ee_pose_l()
    , ee_pose_r()
    , q_exp_l()
    , q_exp_r()
    , cap_set()
    , hand_q_left()
    , hand_q_right()  {
  (void)_alloc;
      ee_pose_l.assign(0.0);

      ee_pose_r.assign(0.0);

      q_exp_l.assign(0.0);

      q_exp_r.assign(0.0);

      cap_set.assign(0.0);

      hand_q_left.assign(0.0);

      hand_q_right.assign(0.0);
  }



   typedef int64_t _time_stamp_type;
  _time_stamp_type time_stamp;

   typedef int32_t _arm_mode_type;
  _arm_mode_type arm_mode;

   typedef boost::array<float, 6>  _ee_pose_l_type;
  _ee_pose_l_type ee_pose_l;

   typedef boost::array<float, 6>  _ee_pose_r_type;
  _ee_pose_r_type ee_pose_r;

   typedef boost::array<float, 7>  _q_exp_l_type;
  _q_exp_l_type q_exp_l;

   typedef boost::array<float, 7>  _q_exp_r_type;
  _q_exp_r_type q_exp_r;

   typedef boost::array<float, 2>  _cap_set_type;
  _cap_set_type cap_set;

   typedef boost::array<float, 6>  _hand_q_left_type;
  _hand_q_left_type hand_q_left;

   typedef boost::array<float, 6>  _hand_q_right_type;
  _hand_q_right_type hand_q_right;





  typedef boost::shared_ptr< ::sdk::RosControl_<ContainerAllocator> > Ptr;
  typedef boost::shared_ptr< ::sdk::RosControl_<ContainerAllocator> const> ConstPtr;

}; // struct RosControl_

typedef ::sdk::RosControl_<std::allocator<void> > RosControl;

typedef boost::shared_ptr< ::sdk::RosControl > RosControlPtr;
typedef boost::shared_ptr< ::sdk::RosControl const> RosControlConstPtr;

// constants requiring out of line definition



template<typename ContainerAllocator>
std::ostream& operator<<(std::ostream& s, const ::sdk::RosControl_<ContainerAllocator> & v)
{
ros::message_operations::Printer< ::sdk::RosControl_<ContainerAllocator> >::stream(s, "", v);
return s;
}


template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator==(const ::sdk::RosControl_<ContainerAllocator1> & lhs, const ::sdk::RosControl_<ContainerAllocator2> & rhs)
{
  return lhs.time_stamp == rhs.time_stamp &&
    lhs.arm_mode == rhs.arm_mode &&
    lhs.ee_pose_l == rhs.ee_pose_l &&
    lhs.ee_pose_r == rhs.ee_pose_r &&
    lhs.q_exp_l == rhs.q_exp_l &&
    lhs.q_exp_r == rhs.q_exp_r &&
    lhs.cap_set == rhs.cap_set &&
    lhs.hand_q_left == rhs.hand_q_left &&
    lhs.hand_q_right == rhs.hand_q_right;
}

template<typename ContainerAllocator1, typename ContainerAllocator2>
bool operator!=(const ::sdk::RosControl_<ContainerAllocator1> & lhs, const ::sdk::RosControl_<ContainerAllocator2> & rhs)
{
  return !(lhs == rhs);
}


} // namespace sdk

namespace ros
{
namespace message_traits
{





template <class ContainerAllocator>
struct IsMessage< ::sdk::RosControl_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsMessage< ::sdk::RosControl_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::sdk::RosControl_<ContainerAllocator> >
  : TrueType
  { };

template <class ContainerAllocator>
struct IsFixedSize< ::sdk::RosControl_<ContainerAllocator> const>
  : TrueType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sdk::RosControl_<ContainerAllocator> >
  : FalseType
  { };

template <class ContainerAllocator>
struct HasHeader< ::sdk::RosControl_<ContainerAllocator> const>
  : FalseType
  { };


template<class ContainerAllocator>
struct MD5Sum< ::sdk::RosControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "db39f8ec46b9260a112172ed9cbb65b6";
  }

  static const char* value(const ::sdk::RosControl_<ContainerAllocator>&) { return value(); }
  static const uint64_t static_value1 = 0xdb39f8ec46b9260aULL;
  static const uint64_t static_value2 = 0x112172ed9cbb65b6ULL;
};

template<class ContainerAllocator>
struct DataType< ::sdk::RosControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "sdk/RosControl";
  }

  static const char* value(const ::sdk::RosControl_<ContainerAllocator>&) { return value(); }
};

template<class ContainerAllocator>
struct Definition< ::sdk::RosControl_<ContainerAllocator> >
{
  static const char* value()
  {
    return "int64 time_stamp\n"
"int32 arm_mode\n"
"float32[6] ee_pose_l\n"
"float32[6] ee_pose_r\n"
"float32[7] q_exp_l\n"
"float32[7] q_exp_r\n"
"float32[2] cap_set\n"
"float32[6] hand_q_left\n"
"float32[6] hand_q_right\n"
;
  }

  static const char* value(const ::sdk::RosControl_<ContainerAllocator>&) { return value(); }
};

} // namespace message_traits
} // namespace ros

namespace ros
{
namespace serialization
{

  template<class ContainerAllocator> struct Serializer< ::sdk::RosControl_<ContainerAllocator> >
  {
    template<typename Stream, typename T> inline static void allInOne(Stream& stream, T m)
    {
      stream.next(m.time_stamp);
      stream.next(m.arm_mode);
      stream.next(m.ee_pose_l);
      stream.next(m.ee_pose_r);
      stream.next(m.q_exp_l);
      stream.next(m.q_exp_r);
      stream.next(m.cap_set);
      stream.next(m.hand_q_left);
      stream.next(m.hand_q_right);
    }

    ROS_DECLARE_ALLINONE_SERIALIZER
  }; // struct RosControl_

} // namespace serialization
} // namespace ros

namespace ros
{
namespace message_operations
{

template<class ContainerAllocator>
struct Printer< ::sdk::RosControl_<ContainerAllocator> >
{
  template<typename Stream> static void stream(Stream& s, const std::string& indent, const ::sdk::RosControl_<ContainerAllocator>& v)
  {
    s << indent << "time_stamp: ";
    Printer<int64_t>::stream(s, indent + "  ", v.time_stamp);
    s << indent << "arm_mode: ";
    Printer<int32_t>::stream(s, indent + "  ", v.arm_mode);
    s << indent << "ee_pose_l[]" << std::endl;
    for (size_t i = 0; i < v.ee_pose_l.size(); ++i)
    {
      s << indent << "  ee_pose_l[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.ee_pose_l[i]);
    }
    s << indent << "ee_pose_r[]" << std::endl;
    for (size_t i = 0; i < v.ee_pose_r.size(); ++i)
    {
      s << indent << "  ee_pose_r[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.ee_pose_r[i]);
    }
    s << indent << "q_exp_l[]" << std::endl;
    for (size_t i = 0; i < v.q_exp_l.size(); ++i)
    {
      s << indent << "  q_exp_l[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.q_exp_l[i]);
    }
    s << indent << "q_exp_r[]" << std::endl;
    for (size_t i = 0; i < v.q_exp_r.size(); ++i)
    {
      s << indent << "  q_exp_r[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.q_exp_r[i]);
    }
    s << indent << "cap_set[]" << std::endl;
    for (size_t i = 0; i < v.cap_set.size(); ++i)
    {
      s << indent << "  cap_set[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.cap_set[i]);
    }
    s << indent << "hand_q_left[]" << std::endl;
    for (size_t i = 0; i < v.hand_q_left.size(); ++i)
    {
      s << indent << "  hand_q_left[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.hand_q_left[i]);
    }
    s << indent << "hand_q_right[]" << std::endl;
    for (size_t i = 0; i < v.hand_q_right.size(); ++i)
    {
      s << indent << "  hand_q_right[" << i << "]: ";
      Printer<float>::stream(s, indent + "  ", v.hand_q_right[i]);
    }
  }
};

} // namespace message_operations
} // namespace ros

#endif // SDK_MESSAGE_ROSCONTROL_H
