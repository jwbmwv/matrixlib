# Documentation

This directory contains MatrixLib's comprehensive documentation.

## Documentation Files

### User Documentation
- **[README.md](../README.md)** - Project overview and quick start
- **[QUICK_REFERENCE.md](../QUICK_REFERENCE.md)** - Compact API cheat sheet  
- **[QUICKSTART.md](../QUICKSTART.md)** - Getting started guide
- **[MIGRATION.md](../MIGRATION.md)** - Migrating from Eigen, GLM, or custom code
- **[PERFORMANCE.md](../PERFORMANCE.md)** - Benchmark results and comparisons

### Technical Documentation
- **[API_Documentation.md](API_Documentation.md)** - Complete API reference
- **[Cpp_Standard_Optimizations.md](Cpp_Standard_Optimizations.md)** - C++ feature detection
- **[Sanitizer_Safety.md](Sanitizer_Safety.md)** - Memory safety and UB prevention
- **[IAR_Integration.md](IAR_Integration.md)** - IAR Embedded Workbench usage

### Generated Documentation
- **[doxygen/](doxygen/)** - Doxygen HTML output (run `doxygen` to generate)

## Design Documentation

Architecture diagrams in PlantUML format:

- **[design/class_hierarchy.puml](../design/class_hierarchy.puml)** - Class relationships
- **[design/module_structure.puml](../design/module_structure.puml)** - Header organization
- **[design/cpp_standard_features.puml](../design/cpp_standard_features.puml)** - C++ features
- **[design/memory_layout.puml](../design/memory_layout.puml)** - Memory alignment
- **[design/safety_architecture.puml](../design/safety_architecture.puml)** - Safety mechanisms
- **[design/simd_optimization.puml](../design/simd_optimization.puml)** - SIMD optimizations
- **[design/transformation_pipeline.puml](../design/transformation_pipeline.puml)** - Transform flow

Generate SVG diagrams:
```bash
cd design
java -jar plantuml.jar -tsvg *.puml
```

## Building Documentation

### Generate Doxygen Docs

```bash
# Install doxygen
sudo apt install doxygen graphviz  # Ubuntu
brew install doxygen graphviz      # macOS

# Build
mkdir build && cd build
cmake .. -DMATRIX_LINEAR_BUILD_DOCS=ON
make docs

# View
open ../docs/doxygen/html/index.html
```

### Documentation Coverage

| Component | Status | Location |
|-----------|--------|----------|
| Vec | ✅ Complete | [API_Documentation.md](API_Documentation.md#vec-class) |
| Mat | ✅ Complete | [API_Documentation.md](API_Documentation.md#mat-class) |
| Quaternion | ✅ Complete | [API_Documentation.md](API_Documentation.md#quaternion-class) |
| Constants | ✅ Complete | [API_Documentation.md](API_Documentation.md#constants-namespace) |
| Examples | ✅ Complete | [../examples/](../examples/) |
| Benchmarks | ✅ Complete | [../benchmarks/](../benchmarks/) |
| Tests | ✅ Complete | [../tests/](../tests/) |

## Contributing to Documentation

### Adding New Documentation

1. **API Changes**: Update [API_Documentation.md](API_Documentation.md)
2. **New Features**: Add to [README.md](../README.md) and [QUICK_REFERENCE.md](../QUICK_REFERENCE.md)
3. **Performance**: Update [PERFORMANCE.md](../PERFORMANCE.md) with benchmarks
4. **Examples**: Add to [../examples/](../examples/) with inline documentation

### Documentation Style Guide

- Use Markdown for text documentation
- Add inline code examples with expected output
- Include both C++11 and modern C++ variants where relevant
- Provide real-world use cases
- Keep Quick Reference concise (< 1 page per section)

### Doxygen Comments

Use Javadoc style in headers:
```cpp
/// @brief Brief description
/// @details Detailed explanation
/// @param name Parameter description
/// @return Return value description
/// @note Additional notes
/// @warning Warnings about edge cases
```

## Documentation Roadmap

- [x] API Documentation
- [x] Quick Reference
- [x] Migration Guide
- [x] Performance Benchmarks
- [x] Examples (Basic, Sensor Fusion, Kinematics, Graphics)
- [x] Doxygen Configuration
- [ ] Video tutorials
- [ ] Interactive API explorer
- [ ] Jupyter notebook examples

## Questions?

- 📖 Check the [FAQ](API_Documentation.md#faq) section
- 💬 Ask in [GitHub Issues](https://github.com/yourusername/matrixlib/issues)
- 📧 Email: your.email@example.com
