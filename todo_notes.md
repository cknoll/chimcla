# chimcla TODO notes

- [ ] improve handling of `file-info.sqlite`
    - Problem: file should not live in Repo but some unit tests depend on it
    - Unittests might change production data
- [ ] automate stage 1:
    - [ ] resize
    - [ ] empty slot detection
    - [ ] incomplete form detection
    - [ ] shading correction
    - [ ] parallelization
- [ ] test stage 1
    - [x] create png-images from jpg-templates
