// pybind surface of the gemm module: every py:: spelling in the family lives
// here — the None-tolerant argument marshalling, the dict shapes of the
// planner's introspection, and the module registration. The typed C++ face is
// gemm/api.h; gemm.cu holds the implementations.
//
// Dictionaries are the wire here, in both directions: the state report and the
// config patch. Each key set is spelled exactly once — the report's keys below,
// the patch's in `kPatchKeys` — next to the struct they mirror, because
// csrc/bench's dispatch_grid / model_capture / diff_rows / tune_plan_table read
// those keys and astrai.extension.plan writes them.

#include <torch/extension.h>

#include <stdexcept>
#include <string>
#include <vector>

#include "common/device.cuh"
#include "gemm/api.h"
#include "gemm/plan_table.h"

namespace astrai {
namespace fp8 {
void bind_fp8(py::module& m);
}  // namespace fp8

namespace gemm {
namespace {

// py::object -> torch::Tensor with a uniform error message; a none object
// stays undefined (callers gate on is_none()).
torch::Tensor cast_tensor_arg(const py::object& o, const char* name) {
    try {
        return o.cast<torch::Tensor>();
    } catch (const py::cast_error&) {
        TORCH_CHECK(false, name, " must be a torch.Tensor or None");
        return {};
    }
}

// pybind surface: None-tolerant operand scales and bias (``cast_tensor_arg``
// keeps the "must be a torch.Tensor or None" message), then the shared
// implementation the composed fp8 linear also calls (see gemm/api.h).
torch::Tensor quant_gemm(torch::Tensor a, torch::Tensor b, py::object a_scale,
                         py::object b_scale, bool trans_a, bool trans_b,
                         py::object bias) {
    auto opt = [](const py::object& o,
                  const char* name) -> c10::optional<torch::Tensor> {
        if (o.is_none()) return c10::nullopt;
        return cast_tensor_arg(o, name);
    };
    return quant_gemm_impl(a, b, opt(a_scale, "a_scale"),
                           opt(b_scale, "b_scale"), trans_a, trans_b,
                           opt(bias, "bias"));
}

// ---------------------------------------------------------------------------
// Marshal the typed planner surface into the dict shapes the Python tools
// read. One key list per struct, no second copy anywhere.
// ---------------------------------------------------------------------------

py::dict probe_dict(const PlanProbe& r) {
    py::dict d;
    d["source"] = r.source;
    d["cta"] = r.cta;
    d["stages"] = r.stages;
    d["raster"] = r.raster;
    d["kk"] = r.kk;
    d["perf_class"] = r.perf_class;
    d["crosswise"] = r.crosswise;
    return d;
}

py::dict config_dict(const GemmConfigState& s) {
    py::dict d, table, staging;
    d["planner"] = s.planner;
    d["planner_mode"] = s.planner_mode;
    d["log"] = s.log;
    table["off"] = s.table_off;
    table["override_rows"] = s.override_rows;
    table["override_source"] = s.override_source;
    table["injected_rows"] = s.injected_rows;
    table["injected_source"] = s.injected_source;
    d["table"] = table;
    staging["tma"] = s.staging_tma;
    staging["mx"] = s.staging_mx;
    d["staging"] = staging;
    return d;
}

py::dict facts_dict() {
    const DeviceFacts dev = astrai::device_facts();
    py::dict d;
    d["sms"] = dev.sms;
    d["smem_max"] = dev.smem_max;
    d["smem_per_sm"] = dev.smem_per_sm;
    d["regs_per_sm"] = dev.regs_per_sm;
    d["l2_bytes"] = dev.l2_bytes;
    d["cc"] = dev.cc;
    return d;
}

// A patch arrives as one dict, and a key that is absent leaves its knob alone —
// that is the whole contract. The dict is the point: with a positional parameter
// list this file had to spell the same seven names three times (the converter's
// parameters, its conversion bodies, and the registration's py::arg list) on top
// of gemm/api.h's struct and astrai.extension.plan's ``configure`` signature.
// The table below is the C++ half of that vocabulary; those other two are the
// typed and the documented ends.

// A str planner is the binding's spelling of the mode: "" restores the unset
// state (the env seed decides, and hybrid is what an unseeded process resolves
// to), a name selects a mode, an int passes straight through for configure() to
// range-check.
void patch_planner(GemmConfigPatch& patch, const py::object& value) {
    if (py::isinstance<py::str>(value)) {
        const std::string name = value.cast<std::string>();
        if (name.empty()) {
            patch.planner_mode = -1;
            return;
        }
        int mode = -1;
        if (!parse_planner_mode(name, mode))
            throw std::invalid_argument(
                "planner must be 'table', 'hybrid' or 'model', got '" + name +
                "'");
        patch.planner_mode = mode;
        return;
    }
    patch.planner_mode = value.cast<int>();
}

// ``rows`` is a row-file path or inline row text; ``tier`` names which row
// source it addresses ("override", the experimenter's, is the default).
void patch_rows(GemmConfigPatch& patch, const py::object& value) {
    patch.rows = value.cast<std::string>();
}

void patch_tier(GemmConfigPatch& patch, const py::object& value) {
    const std::string name = value.cast<std::string>();
    if (name == "override") {
        patch.tier = RowTier::Override;
        return;
    }
    if (name == "injected") {
        patch.tier = RowTier::Injected;
        return;
    }
    throw std::invalid_argument("tier must be 'override' or 'injected', got '" +
                                name + "'");
}

struct PatchKey {
    const char* key;
    void (*apply)(GemmConfigPatch&, const py::object&);
};

const PatchKey kPatchKeys[] = {
    {"planner", patch_planner},
    {"log",
     [](GemmConfigPatch& p, const py::object& v) { p.log = v.cast<bool>(); }},
    {"tma",
     [](GemmConfigPatch& p, const py::object& v) {
         p.staging_tma = v.cast<bool>();
     }},
    {"mx",
     [](GemmConfigPatch& p, const py::object& v) {
         p.staging_mx = v.cast<bool>();
     }},
    {"table_off",
     [](GemmConfigPatch& p, const py::object& v) {
         p.table_off = v.cast<bool>();
     }},
    {"rows", patch_rows},
    {"tier", patch_tier},
};

std::string patch_keys() {
    std::string out;
    for (const PatchKey& key : kPatchKeys) {
        if (!out.empty()) out += ", ";
        out += key.key;
    }
    return out;
}

GemmConfigPatch patch_from(const py::dict& patch) {
    GemmConfigPatch out;
    for (const auto& item : patch) {
        const py::object key_object = py::reinterpret_borrow<py::object>(item.first);
        TORCH_CHECK(py::isinstance<py::str>(key_object),
                    "gemm config keys must be strings; the plan's knobs are ",
                    patch_keys());
        const std::string key = key_object.cast<std::string>();
        bool known = false;
        for (const PatchKey& candidate : kPatchKeys) {
            if (key != candidate.key) continue;
            candidate.apply(out, py::reinterpret_borrow<py::object>(item.second));
            known = true;
            break;
        }
        TORCH_CHECK(known, "unknown gemm config key '", key,
                    "'; the plan's knobs are ", patch_keys());
    }
    return out;
}

py::dict probe_binding(int64_t m, int64_t n, int64_t k, at::ScalarType dt_a,
                       at::ScalarType dt_b, bool trans_a, bool trans_b,
                       int64_t batch) {
    return probe_dict(plan_probe(m, n, k, dt_a, dt_b, trans_a, trans_b, batch));
}

py::dict configure_binding(const py::dict& patch) {
    return config_dict(configure(patch_from(patch)));
}

py::dict config_state_binding() { return config_dict(config_state()); }

}  // namespace
}  // namespace gemm
}  // namespace astrai

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // The fp8 training linear (forward + backward) lives in this module:
    // its composition launches through the GEMM dispatch below, whose
    // plan table / planner state must stay single-source.
    astrai::fp8::bind_fp8(m);
    m.def("quant_gemm", &astrai::gemm::quant_gemm, py::arg("a"), py::arg("b"),
          py::arg("a_scale") = py::none(), py::arg("b_scale") = py::none(),
          py::arg("trans_a") = false, py::arg("trans_b") = true,
          py::arg("bias") = py::none());
    m.def("plan_probe", &astrai::gemm::probe_binding, py::arg("m"),
          py::arg("n"), py::arg("k"), py::arg("dt_a"), py::arg("dt_b"),
          py::arg("trans_a") = false, py::arg("trans_b") = true,
          py::arg("batch") = 1);
    m.def("configure", &astrai::gemm::configure_binding, py::arg("patch"),
          "Apply a config patch (a dict of the plan's knobs) and return the "
          "resulting state");
    // The patch-dict schema, so a capability probe can tell a stale build from
    // a current one: ``configure``'s old keyword signature is not
    // distinguishable by hasattr, only by calling it.
    m.attr("CONFIG_API") = 2;
    m.def("config_state", &astrai::gemm::config_state_binding);
    m.def("tile_class_names", &astrai::gemm::tile_class_names);
    m.def("tile_vocabulary", &astrai::gemm::tile_vocabulary);
    m.def("device_facts_info", &astrai::gemm::facts_dict);
}
