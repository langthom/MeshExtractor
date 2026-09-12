#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <type_traits>
#include <variant>
#include <vector>

namespace parallel_mesh_extractor {

  // Enumeration listing the supported voxel types.
  enum class VoxelType : std::uint32_t {
     INT8 = 7,  INT16 = 15,  INT32 = 31,  INT64 = 63, FLOAT32 = 24,
    UINT8 = 8, UINT16 = 16, UINT32 = 32, UINT64 = 64, FLOAT64 = 53,
  };


  // Helper template mapping VoxelType enum values to C++ types.
  template<VoxelType T> struct VoxelTypeToType;
  template<> struct VoxelTypeToType<VoxelType::INT8   > { using type = std::int8_t;   };
  template<> struct VoxelTypeToType<VoxelType::INT16  > { using type = std::int16_t;  };
  template<> struct VoxelTypeToType<VoxelType::INT32  > { using type = std::int32_t;  };
  template<> struct VoxelTypeToType<VoxelType::INT64  > { using type = std::int64_t;  };
  template<> struct VoxelTypeToType<VoxelType::UINT8  > { using type = std::uint8_t;  };
  template<> struct VoxelTypeToType<VoxelType::UINT16 > { using type = std::uint16_t; };
  template<> struct VoxelTypeToType<VoxelType::UINT32 > { using type = std::uint32_t; };
  template<> struct VoxelTypeToType<VoxelType::UINT64 > { using type = std::uint64_t; };
  template<> struct VoxelTypeToType<VoxelType::FLOAT32> { using type = float;         };
  template<> struct VoxelTypeToType<VoxelType::FLOAT64> { using type = double;        };

 // Template for describing a list of VoxelType values at compile time.
  template<VoxelType... Types>
  using VoxelTypeSelection = std::integer_sequence<VoxelType, Types...>;

  // Default type list containing *all* supported voxel types.
  using ALL_VOXEL_TYPES = VoxelTypeSelection<
    VoxelType::INT8,  VoxelType::INT16,  VoxelType::INT32,  VoxelType::INT64,  VoxelType::FLOAT32,
    VoxelType::UINT8, VoxelType::UINT16, VoxelType::UINT32, VoxelType::UINT64, VoxelType::FLOAT64
  >;

  // A simple type tag encapsulating a VoxelType enumeration value and its according C++ type.
  template<VoxelType E>
  struct VoxelTypeTag {
    static constexpr VoxelType value = E;
    using type = typename VoxelTypeToType<E>::type;
  };

  // Helper template which takes a compile-time list of supported voxel types.
  // It unpacks that sequence (yielding the variadic type pack), making the individual types usable.
  // During compilation, the contents of its "create" function consider a fold expression which
  // stages a comparison which compares the requested VoxelType against all passed supported
  // VoxelTypes. If the given voxel type matches one of these, an std::optional is created containing
  // a VoxelTypeTag instance encapsulating the information of the matched type. Otherwise, it
  // returns std::nullopt.
  template<class Sequence>
  struct MakeSelectedVariantHelper;

  template<VoxelType... SelectedTypes>
  struct MakeSelectedVariantHelper<VoxelTypeSelection<SelectedTypes...>> {
    static constexpr auto create(VoxelType runtimeVoxelType)
      -> std::optional<std::variant<VoxelTypeTag<SelectedTypes>...>>
    {
      using VariantType = std::variant<VoxelTypeTag<SelectedTypes>...>;
      std::optional<VariantType> result = std::nullopt;

      ((runtimeVoxelType == SelectedTypes
          ? (result = VariantType{VoxelTypeTag<SelectedTypes>{}}, true)
          : false)
       || ...);

      return result;
    };
  };


  // Helper function which takes a VoxelTypeSelection type list, splits it open, and stores
  // the first VoxelType value for later retrieval.
  template<class Sequence>
  struct GetFirstVoxelType;

  template<VoxelType First, VoxelType... Rest>
  struct GetFirstVoxelType<VoxelTypeSelection<First, Rest...>> {
    static constexpr VoxelType value = First;
  };


  // Main dispatching function.
  // Takes the runtime voxel type, a functor (which is a special lambda taking a type tag) and
  // the function arguments. Dispatches and invokes the underlying function.
  template<class VoxelTypeSelection, class Functor, class... Args>
  decltype(auto) dispatchOverVoxelType(VoxelType voxelType, Functor&& functor, Args&&... args) {
    // Get the first VoxelType in the allowed list to determine the return type.
    constexpr VoxelType FirstEnum = GetFirstVoxelType<VoxelTypeSelection>::value;
    using ReturnType = decltype(functor(VoxelTypeTag<FirstEnum>{}, std::forward<Args>(args)...));

    // Generate an optional type variant containing the type tag (dispatched).
    auto targetVariant = MakeSelectedVariantHelper<VoxelTypeSelection>::create(voxelType);

    // If there is a match, we visit the variant with a custom lambda expression, which receives
    // the type tag stored in the variant. Note that it only contains exactly one tag which is
    // extracted by the MakeSelectedVariantHelper helper template. Once the lambda gets applied,
    // it unpacks the variant tag and calls the underlying binary functor, forwarding its arguments.
    if (targetVariant.has_value()) {
      return std::visit([&](auto tag) -> decltype(auto) {
              static constexpr VoxelType E = decltype(tag)::value;
              return functor(VoxelTypeTag<E>{}, std::forward<Args>(args)...);
          },
          *targetVariant);
    }

    if constexpr(std::is_void_v<ReturnType>) {
      return;
    } else {
      return ReturnType{};
    }
  }

// Helper dispatch macro, which dispatches the given runtime "voxelType", a given "function"
// which is templated on the VoxelType enumeration, and a variadic number of arguments.
// Also, you pass in the "VoxelTypeSelection" which defines a list of admissible VoxelType
// instances. For convenience, an ALL_VOXEL_TYPES list is pre-defined.
// It invokes the main dispatching function, passing a custom lambda function that receives a type
// tag storing the runtime type associated to the requested runtime "voxelType", and calling the
// templated function with that type.
#define DISPATCH_VOXEL_TYPE(VoxelTypeSelection, voxelType, function, ...)                       \
  dispatchOverVoxelType<VoxelTypeSelection>(                                                    \
    voxelType,                                                                                  \
    [](auto tag, auto&&... args) {                                                              \
      using T = typename decltype(tag)::type;                                                   \
      return function<T>(std::forward<decltype(args)>(args)...);                                \
    },                                                                                          \
    ##__VA_ARGS__                                                                               \
  )


  /// Structure defining the meta data of volumetric datasets.
  struct VolumeMetaData {
    /// Dimensions of the voxel dataset, in (X, Y, Z), i.e.,
    /// in (column, row, slice) as continuous block in memory.
    std::array<std::uint32_t, 3> dim;

    /// Voxel size in [mm] along its X, Y, Z axes.
    std::array<float, 3> spacing;

    /// World coordinate of the very first voxel, in the same unit as the spacing. Everything
    /// extracted from the volume is translated by it, so that a mesh lands where the scan says it
    /// belongs rather than at the corner of the voxel grid. Defaulted, because a volume that names
    /// no origin simply sits at zero.
    std::array<float, 3> origin{0.0f, 0.0f, 0.0f};

    /// Enumeration value defining which voxel type the natively stored data takes.
    VoxelType voxelType;

    std::array<std::size_t, 3> ComputeOffsets(void) const;

    size_t GetElementSizeInBytes(void) const;

    std::uintmax_t GetNumberOfVoxels(void) const;
  };

} // namespace parallel_mesh_extractor

