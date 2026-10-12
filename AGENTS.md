
## Implementation style (when implementing a plan, not when planning)
- Think briefly, decide, then act. Do not draft file contents in your reasoning; put the content directly into the write or edit tool call.
- One tool call per turn. Write or edit a single file, wait for the result, then do the next one.
- Work through the plan one step at a time and say which step you are on.
- Keep files small. For any file likely to exceed about 150 lines, first write a skeleton with section markers (like `<!-- SECTION: name -->`), then fill one section per turn by replacing its marker with edit.
- Long prompt or documentation files: write the outline first, then each section separately.
