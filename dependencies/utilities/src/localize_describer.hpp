#pragma once

/**
 * @file
 * @brief Reading the describer out of a matches directory's
 *        `image_describer.json`.
 *
 * One function in its own translation unit, and its own C++17 library target,
 * for one reason: openMVG's bundled cereal does not compile under C++20. Its
 * `StaticObject::LockGuard` declares a defaulted copy constructor, which C++20
 * (P1008) stopped treating as leaving an aggregate behind, so cereal's own
 * `return LockGuard{};` no longer finds a constructor. `pgs-localize` is C++20
 * because bvh needs `std::span`; this file is the one place that has to meet
 * cereal, so it is the one place held back.
 *
 * Nothing cereal crosses this header. The returned describer is an openMVG type
 * compiled identically on both sides.
 */

#include <filesystem>
#include <memory>

#include <openMVG/features/image_describer.hpp>

namespace pgs::localize
{

/**
 * @brief Load the describer recorded in a matches directory.
 *
 * The query has to be described the way the database was -- the descriptor type
 * and its parameters both -- so this is read from the scene's own record of it
 * and is never a flag.
 *
 * @throws std::runtime_error naming the file when it cannot be read or names
 * no describer.
 */
auto LoadDescriber(const std::filesystem::path& imageDescriberJson)
    -> std::unique_ptr<openMVG::features::Image_describer>;

}  // namespace pgs::localize
