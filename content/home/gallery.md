---
# An instance of the Blank widget with a Gallery page element.
# Documentation: https://wowchemy.com/docs/getting-started/page-builder/
widget: markdown

# This file represents a page section.
headless: true

# Order that this section appears on the page.
weight: 120

title: Gallery
subtitle: Some of my traditional Chinese paintings

design:
  columns: '2'
---

{{< gallery album="paintings" sortOrder="desc" thumbnailResizeOptions="200x200 q90 Lanczos" showExif=true previewType="blur" embedPreview=true loadJQuery=true >}}
