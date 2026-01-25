# MatrixLib Design Documentation

This directory contains UML diagrams documenting the architecture, design patterns, and implementation details of MatrixLib.

## Diagrams

### 1. Class Hierarchy (`class_hierarchy.puml`)
- **Purpose**: Shows the inheritance and composition relationships between core classes
- **Content**:
  - `Vec<T, N>` template class
  - `Mat<T, R, C>` generic matrix
  - `SquareMat<T, N>` and specializations (2×2, 3×3, 4×4)
  - `Quaternion<T>` rotation class
  - Type aliases (Vec2f, Mat3f, Quatf, etc.)
  - Relationships and dependencies
  - SIMD compatibility notes

### 2. Module Structure (`module_structure.puml`)
- **Purpose**: Illustrates the modular header architecture and dependencies
- **Content**:
  - Core headers (compiler_features, constants, vector, matrix, quaternion)
  - Main header (matrixlib.hpp)
  - Type alias headers (vec2D, vec3D, matrix2D, matrix3D)
  - Build system integration (CMake, Zephyr)
  - SIMD backend options (NEON, CMSIS-DSP)
  - Testing infrastructure
  - Documentation organization

### 3. Memory Layout (`memory_layout.puml`)
- **Purpose**: Documents data structure memory layouts and alignment
- **Content**:
  - Byte-level memory layout for Vec3, Mat3, Mat4, Quaternion
  - 16-byte alignment details
  - Row-major storage explanation
  - SIMD register mapping (NEON, CMSIS-DSP)
  - Trivially copyable verification
  - Padding and alignment guarantees

### 4. Transformation Pipeline (`transformation_pipeline.puml`)
- **Purpose**: Sequence diagram showing typical 3D transformation workflows
- **Content**:
  - Object-to-world space transformations
  - Quaternion rotation operations
  - Camera transform setup (look_at)
  - Interpolation (lerp, slerp)
  - Matrix composition
  - Point vs direction transformation

### 5. SIMD Optimization (`simd_optimization.puml`)
- **Purpose**: Details the three-tier SIMD optimization strategy
- **Content**:
  - Tier 1: ARM NEON (Cortex-A, ARM64, Apple Silicon)
  - Tier 2: CMSIS-DSP (Cortex-M with FPU)
  - Tier 3: Generic C++ (all platforms)
  - Performance characteristics by tier
  - Optimization selection logic
  - Intrinsics used per tier
  - Enable flags and detection

### 6. C++ Standard Features (`cpp_standard_features.puml`)
- **Purpose**: Shows progressive feature adoption from C++11 to C++26
- **Content**:
  - C++11 baseline features (required)
  - C++14 extensions (relaxed constexpr)
  - C++17 additions (if constexpr, [[nodiscard]])
  - C++20 features (std::bit_cast, concepts, consteval)
  - C++23 enhancements (if consteval)
  - C++26 capabilities (constexpr trigonometry)
  - Compiler support matrix
  - Performance impact per standard

### 7. Safety Architecture (`safety_architecture.puml`)
- **Purpose**: Documents type safety mechanisms and UB prevention strategies
- **Content**:
  - MATRIX_BIT_CAST implementation (C++11 vs C++20)
  - Quaternion vec() safety (return by value)
  - SIMD type-guarded casts
  - std::numeric_limits<T>::epsilon() usage
  - std::is_trivially_copyable verification
  - Sanitizer testing (UBSan, ASan)
  - Edge case test coverage
  - Documentation references

## Viewing the Diagrams

### Online Viewers
- [PlantUML Online Server](http://www.plantuml.com/plantuml/uml/)
- [PlantText](https://www.planttext.com/)
- Copy and paste the `.puml` file contents into the editor

### VS Code Extensions
- **PlantUML** by jebbs
  - Install from Extensions marketplace
  - Open `.puml` file
  - Press `Alt+D` to preview
  - Export to PNG/SVG with `Ctrl+Shift+E`

### Command-Line Tool
```bash
# Install PlantUML
sudo apt-get install plantuml  # Ubuntu/Debian
brew install plantuml          # macOS

# Generate PNG
plantuml design/class_hierarchy.puml

# Generate SVG (recommended for documentation)
plantuml -tsvg design/class_hierarchy.puml

# Generate all diagrams
plantuml design/*.puml
```

### Docker
```bash
# Generate all diagrams to PNG
docker run -v $(pwd):/data plantuml/plantuml design/*.puml

# Generate SVG
docker run -v $(pwd):/data plantuml/plantuml -tsvg design/*.puml
```

## Export Formats

PlantUML supports multiple output formats:

| Format | Extension | Best For |
|--------|-----------|----------|
| PNG | `.png` | Quick viewing, GitHub README |
| SVG | `.svg` | Scalable, high quality, web |
| PDF | `.pdf` | Documentation, printing |
| EPS | `.eps` | LaTeX, academic papers |
| LaTeX | `.tex` | Academic publications |

**Recommended**: Use SVG for documentation (scales perfectly, small file size)

## Generating All Diagrams

### Batch Generation Script (Linux/macOS)
```bash
#!/bin/bash
cd design
for file in *.puml; do
    plantuml -tsvg "$file"
    plantuml -tpng "$file"
done
```

### Batch Generation Script (Windows PowerShell)
```powershell
cd design
Get-ChildItem *.puml | ForEach-Object {
    plantuml -tsvg $_.FullName
    plantuml -tpng $_.FullName
}
```

### Makefile
```makefile
.PHONY: diagrams
diagrams:
	plantuml -tsvg design/*.puml
	plantuml -tpng design/*.puml
```

## Integration with Documentation

Generated diagrams can be included in Markdown documentation:

```markdown
## Architecture

![Class Hierarchy](design/class_hierarchy.svg)

## Module Structure

![Modules](design/module_structure.svg)
```

## Maintenance

When updating the library:

1. **Code changes → Update diagrams**
   - New classes: Update `class_hierarchy.puml`
   - New modules: Update `module_structure.puml`
   - Performance changes: Update `simd_optimization.puml`

2. **Version documentation**
   - Add date and version notes to diagrams
   - Keep diagrams in sync with code

3. **Export updated diagrams**
   - Run generation script
   - Commit both `.puml` and generated images

## PlantUML Syntax Reference

- [Official Documentation](https://plantuml.com/)
- [Class Diagrams](https://plantuml.com/class-diagram)
- [Component Diagrams](https://plantuml.com/component-diagram)
- [Sequence Diagrams](https://plantuml.com/sequence-diagram)
- [Activity Diagrams](https://plantuml.com/activity-diagram)

## Contributing

When adding new diagrams:

1. Create `.puml` file in `design/` directory
2. Follow existing naming conventions
3. Add title and description
4. Update this README with diagram description
5. Generate SVG/PNG exports
6. Test rendering in multiple viewers

## License

These diagrams are part of MatrixLib and follow the same MIT license.

Copyright (c) 2026 James Baldwin

---

*Generated with PlantUML - Last updated: January 27, 2026*
