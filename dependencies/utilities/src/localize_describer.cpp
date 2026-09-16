#include "localize_describer.hpp"

#include <fstream>
#include <stdexcept>

#include <cereal/archives/json.hpp>

#include <openMVG/features/akaze/image_describer_akaze_io.hpp>
#include <openMVG/features/regions_factory_io.hpp>
#include <openMVG/features/sift/SIFT_Anatomy_Image_Describer_io.hpp>

// The VLFeat-backed describer, which is what pgs-recon's own matches_dir is
// described with (`SIFT_Image_describer` in the recorded JSON). It lives in
// openMVG's nonFree tree and registers a different cereal type from the
// Anatomy one, so both headers have to be here or the archive refuses to
// construct what the file names.
#include <nonFree/sift/SIFT_describer_io.hpp>

namespace fs = std::filesystem;

auto pgs::localize::LoadDescriber(const fs::path& imageDescriberJson)
    -> std::unique_ptr<openMVG::features::Image_describer>
{
  if (not fs::exists(imageDescriberJson)) {
    throw std::runtime_error(
        "no image_describer.json at " + imageDescriberJson.string() +
        "; the query cannot be described the way the scene's regions were");
  }
  std::ifstream stream(imageDescriberJson, std::ios::binary);
  if (not stream.is_open()) {
    throw std::runtime_error("could not read " + imageDescriberJson.string());
  }
  std::unique_ptr<openMVG::features::Image_describer> describer;
  try {
    cereal::JSONInputArchive archive(stream);
    archive(cereal::make_nvp("image_describer", describer));
  } catch (const std::exception& e) {
    throw std::runtime_error("could not read " + imageDescriberJson.string() +
                             ": " + e.what());
  }
  if (not describer) {
    throw std::runtime_error(imageDescriberJson.string() +
                             " named no usable describer");
  }
  return describer;
}
