# project context

we are creating tools for the manufacturing of artaficial ceraimc mosaics using special molds and ceramic toners for decal printing.


# about me

Im an engineer and an entrepreneur developing a new flooring technique


# rules

- act autonomously. do NOT ask for approval or confirmation before editing files, running bash / powershell commands, or applying changes. just do it and report the result.
- do NOT ask "should I proceed?" or "do you want me to continue?" between steps of a task. finish the task, then report.
- exception: still stop and ask if the request is genuinely ambiguous (unclear which of two files, unclear what output format, etc.). "auto-approval" is about mechanics, not intent.
- keep reports and summaries concise — bullet points over paragraphs.
- save output files to output folder.
- never do anything outside the CLAUDE_MOSAIC1.0 folder.
- when you state "I will do N things", the actual run must be exactly N. if the executed count differs (filter expansion, loop iteration count, API call total), stop and reconcile before continuing.
- before any batch or background command, print the explicit resource impact (files affected, API calls, time estimate) and the actual list of items. one-line summaries hide drift.

# project structure

-worflows/ workflow instruction files (plain english recepies for the agent to follow)
-output/ images, csv files, combined files
-resources/ downloaded images etc


