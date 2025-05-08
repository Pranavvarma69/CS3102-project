package main

import (
    "fmt"
    "log"
    "os"
    "os/exec"
    "path/filepath"
    "regexp"
    "sort"
    "strings"
    "time"
    "unicode"

    "github.com/go-git/go-git/v5"
    "github.com/go-git/go-git/v5/plumbing"
    "github.com/go-git/go-git/v5/plumbing/format/diff"
    "github.com/go-git/go-git/v5/plumbing/object"
)

// conventionalCommitRegex matches the basic structure of a conventional commit message.
var conventionalCommitRegex = regexp.MustCompile(`^(\w+)(\([\w-]+\))?:\s.*$`)

// tokenRegex defines characters that act as separators for tokenization.
var tokenRegex = regexp.MustCompile(`[\s():,<>{}.;="*&|!\[\]-]+`)

// defaultIgnoreTypes are types that are pre-selected for ignoring
var defaultIgnoreTypes = []string{"refactor"}

// Represents contribution data for an author.
type authorContribution struct {
    Name       string
    Email      string
    TokenCount int
    CommitCount int
    Files       map[string]int // filename -> tokens
    TypesMap    map[string]int // commit type -> count
}

// CommitInfo stores basic commit information for display and selection
type CommitInfo struct {
    Hash    string
    Message string
    Author  string
    Email   string
}

// CommitAnalysisResult stores the full analysis results
type CommitAnalysisResult struct {
    ConventionalCommits      int
    NonConventionalCommits   int
    SelectedNonConventional  int
    IgnoredTypesCount        map[string]int
    IgnoredTypesTotal        int
    TotalTokens              int
    AuthorContributions      map[string]*authorContribution
    StartDate                time.Time
    EndDate                  time.Time
}

func main() {
    // --- 1. Setup and Open Repository ---
    repoPath := "."
    if len(os.Args) > 1 {
        repoPath = os.Args[1]
    }
    absRepoPath, err := filepath.Abs(repoPath)
    if err != nil {
        log.Fatalf("Error getting absolute path for %s: %v", repoPath, err)
    }

    fmt.Printf("Analyzing Git repository at: %s\n", absRepoPath)

    repo, err := git.PlainOpen(absRepoPath)
    if err != nil {
        log.Fatalf("Error opening repository at %s: %v\nMake sure you are in a valid git repository.", absRepoPath, err)
    }

    // --- 2. Iterate Commits, Collect Types, and Separate Commits ---
    commitTypes := make(map[string]struct{})
    ignoredTypesCount := make(map[string]int)
    conventionalCommits := []*object.Commit{}
    nonConventionalCommits := []*object.Commit{}
    nonConventionalByAuthor := make(map[string][]*object.Commit) // Map email -> non-conventional commits
    nonConventionalInfoList := []CommitInfo{} // For selection UI

    var startDate time.Time
    var endDate time.Time

    refHead, err := repo.Head()
    if err != nil {
        log.Fatalf("Error getting HEAD reference: %v", err)
    }

    cIter, err := repo.Log(&git.LogOptions{From: refHead.Hash()})
    if err != nil {
        log.Fatalf("Error getting commit iterator: %v", err)
    }

    fmt.Println("Scanning commits...")
    err = cIter.ForEach(func(c *object.Commit) error {
        // Track commit dates for analysis timeframe
        if startDate.IsZero() || c.Author.When.Before(startDate) {
            startDate = c.Author.When
        }
        if endDate.IsZero() || c.Author.When.After(endDate) {
            endDate = c.Author.When
        }

        commitType, isValid := parseCommitMessage(c.Message)
        authorKey := strings.ToLower(c.Author.Email)
        if isValid {
            commitTypes[commitType] = struct{}{}
            conventionalCommits = append(conventionalCommits, c)
        } else {
            nonConventionalCommits = append(nonConventionalCommits, c)
            // Group non-conventional commits by author email
            if _, ok := nonConventionalByAuthor[authorKey]; !ok {
                nonConventionalByAuthor[authorKey] = []*object.Commit{}
            }
            nonConventionalByAuthor[authorKey] = append(nonConventionalByAuthor[authorKey], c)

            // Add to info list for selection UI
            nonConventionalInfoList = append(nonConventionalInfoList, CommitInfo{
                Hash:    c.Hash.String()[:7],
                Message: getCommitMessageSummary(c.Message),
                Author:  c.Author.Name,
                Email:   c.Author.Email,
            })
        }
        return nil
    })
    if err != nil {
        log.Fatalf("Error iterating through commits: %v", err)
    }

    if len(conventionalCommits) == 0 && len(nonConventionalCommits) == 0 {
        log.Fatalf("No commits found in the repository.")
    }

    // --- 3. Interactive Filtering with Gum (for Conventional Commits) ---
    ignoredTypes := make(map[string]struct{})
    if len(conventionalCommits) > 0 {
        sortedTypes := prepareCommitTypesForDisplay(commitTypes)

        prompt := "Choose conventional commit types to IGNORE (Space to select, Enter to confirm):"

        // Using modified runGumChoose that allows for 0 selections and pre-selected items
        ignoredTypesSlice, err := runGumChoose(sortedTypes, prompt, true, defaultIgnoreTypes)
        if err != nil {
            if strings.Contains(err.Error(), "gum choose command cancelled") {
                fmt.Println("Selection cancelled. Continuing without ignoring any types.")
            } else {
                log.Fatalf("Error running gum choose: %v\nMake sure 'gum' is installed and in your PATH.", err)
            }
        }

        for _, t := range ignoredTypesSlice {
            ignoredTypes[t] = struct{}{}
        }

        if len(ignoredTypesSlice) > 0 {
            fmt.Printf("Ignoring conventional types: %v\n", ignoredTypesSlice)
        } else {
            fmt.Println("No conventional types selected to ignore.")
        }
    } else {
        fmt.Println("No conventional commits found to filter.")
    }

    // --- 4. Interactive Selection of Non-Conventional Commits to Include ---
    includedNonConventionalCommits := make(map[string]bool)

    if len(nonConventionalCommits) > 0 {
        // Format commit info for display
        nonConventionalOptions := []string{}
        for _, info := range nonConventionalInfoList {
            // Format: [hash] message (author <email>)
            option := fmt.Sprintf("[%s] %s (%s <%s>)",
                info.Hash,
                info.Message,
                info.Author,
                info.Email)
            nonConventionalOptions = append(nonConventionalOptions, option)
        }

        fmt.Println("\nSelect non-conventional commits to manually include:")
        selectedIndices, err := runGumChooseWithIndices(nonConventionalOptions,
            "Choose non-conventional commits to analyze (Space to select, Enter to confirm):",
            true, nil)

        if err != nil {
            if strings.Contains(err.Error(), "gum choose command cancelled") {
                fmt.Println("Selection cancelled. Continuing without including any non-conventional commits.")
            } else {
                log.Printf("Warning: Failed to select non-conventional commits: %v", err)
            }
        }

        // Mark selected commits as included
        for _, idx := range selectedIndices {
            if idx < len(nonConventionalInfoList) {
                commitHash := nonConventionalInfoList[idx].Hash
                includedNonConventionalCommits[commitHash] = true
            }
        }

        if len(selectedIndices) > 0 {
            fmt.Printf("Including %d non-conventional commits for analysis.\n", len(selectedIndices))
        } else {
            fmt.Println("No non-conventional commits selected for inclusion.")
        }
    }

    // --- 5. Analysis of All Commits ---
    result := &CommitAnalysisResult{
        ConventionalCommits:     len(conventionalCommits),
        NonConventionalCommits:  len(nonConventionalCommits),
        SelectedNonConventional: len(includedNonConventionalCommits),
        IgnoredTypesCount:       ignoredTypesCount,
        IgnoredTypesTotal:       0,
        AuthorContributions:     make(map[string]*authorContribution),
        StartDate:               startDate,
        EndDate:                 endDate,
    }

    // Count commits by type that were ignored
    for _, commit := range conventionalCommits {
        commitType, _ := parseCommitMessage(commit.Message)
        if _, shouldIgnore := ignoredTypes[commitType]; shouldIgnore {
            result.IgnoredTypesTotal++
            if _, ok := ignoredTypesCount[commitType]; !ok {
                ignoredTypesCount[commitType] = 0
            }
            ignoredTypesCount[commitType]++
            continue
        }

        // Add to our tracking
        err := calculateCommitTokens(repo, commit, result.AuthorContributions)
        if err != nil {
            log.Printf("Warning: Failed to process conventional commit %s: %v", commit.Hash.String()[:7], err)
        }

        // Track commit type
        authorKey := strings.ToLower(commit.Author.Email)
        if _, ok := result.AuthorContributions[authorKey]; ok {
            commitType, _ := parseCommitMessage(commit.Message)
            if _, ok := result.AuthorContributions[authorKey].TypesMap[commitType]; !ok {
                result.AuthorContributions[authorKey].TypesMap[commitType] = 0
            }
            result.AuthorContributions[authorKey].TypesMap[commitType]++
        }
    }

    // Process only non-conventional commits that were selected for inclusion
    for _, commit := range nonConventionalCommits {
        commitHash := commit.Hash.String()[:7]

        // Skip this commit if we have selections and this commit wasn't selected
        if len(includedNonConventionalCommits) > 0 && !includedNonConventionalCommits[commitHash] {
            continue
        }

        err := calculateCommitTokens(repo, commit, result.AuthorContributions)
        if err != nil {
            log.Printf("Warning: Failed to process non-conventional commit %s: %v", commitHash, err)
            continue
        }
    }

    // Calculate total tokens
    for _, contrib := range result.AuthorContributions {
        result.TotalTokens += contrib.TokenCount
    }

    // --- 6. Display Detailed Combined Results ---
    displayCombinedResults(result, includedNonConventionalCommits, nonConventionalInfoList)
}

// prepareCommitTypesForDisplay sorts commit types and ensures defaultIgnoreTypes are at the top
func prepareCommitTypesForDisplay(commitTypes map[string]struct{}) []string {
    // Create a map to track types we've already added
    typesMap := make(map[string]bool)

    // Extract all types
    allTypes := make([]string, 0, len(commitTypes)+len(defaultIgnoreTypes))
    for t := range commitTypes {
        allTypes = append(allTypes, t)
        typesMap[t] = true
    }

    // Add defaultIgnoreTypes if they don't exist in the commit types
    for _, ignoreType := range defaultIgnoreTypes {
        if !typesMap[ignoreType] {
            allTypes = append(allTypes, ignoreType)
            typesMap[ignoreType] = true
        }
    }

    // Sort alphabetically but keep defaultIgnoreTypes at the top
    sort.Slice(allTypes, func(i, j int) bool {
        // If i is in defaultIgnoreTypes and j is not, i comes first
        iInDefault := contains(defaultIgnoreTypes, allTypes[i])
        jInDefault := contains(defaultIgnoreTypes, allTypes[j])

        if iInDefault && !jInDefault {
            return true
        }
        if !iInDefault && jInDefault {
            return false
        }

        // If both are in defaultIgnoreTypes or neither is, sort alphabetically
        return allTypes[i] < allTypes[j]
    })

    return allTypes
}

// parseCommitMessage checks if a message follows the conventional format
func parseCommitMessage(msg string) (string, bool) {
    lines := strings.SplitN(msg, "\n", 2)
    if len(lines) == 0 {
        return "", false
    }
    matches := conventionalCommitRegex.FindStringSubmatch(lines[0])
    if len(matches) >= 2 {
        return matches[1], true
    }
    return "", false
}

// getCommitMessageSummary returns the first line of the commit message.
func getCommitMessageSummary(msg string) string {
    lines := strings.SplitN(msg, "\n", 2)
    if len(lines) > 0 {
        return strings.TrimSpace(lines[0])
    }
    return ""
}

// calculateCommitTokens analyzes a single commit's diff and adds token counts.
func calculateCommitTokens(repo *git.Repository, commit *object.Commit, contributions map[string]*authorContribution) error {
    var parentCommit *object.Commit
    var err error

    if commit.NumParents() > 0 {
        parentCommit, err = commit.Parent(0)
        if err != nil {
            return fmt.Errorf("could not get parent: %w", err)
        }
    }

    var patch *object.Patch
    if parentCommit != nil {
        patch, err = parentCommit.Patch(commit)
    } else {
        // Initial commit
        commitTree, errTree := commit.Tree()
        if errTree != nil {
            return fmt.Errorf("could not get tree for initial commit: %w", errTree)
        }

        // Try getting empty tree from the repo's object storage
        emptyTree, errEmpty := repo.TreeObject(plumbing.ZeroHash)
        if errEmpty != nil {
            log.Printf("Warning: Could not get empty tree object from repo storage (%v). This might affect initial commit analysis for %s.", errEmpty, commit.Hash.String()[:7])
            return fmt.Errorf("could not get empty tree object: %w", errEmpty)
        }
        patch, err = emptyTree.Patch(commitTree)
    }

    if err != nil {
        return fmt.Errorf("could not get patch: %w", err)
    }

    authorKey := strings.ToLower(commit.Author.Email)
    if _, ok := contributions[authorKey]; !ok {
        contributions[authorKey] = &authorContribution{
            Name:       commit.Author.Name,
            Email:      commit.Author.Email,
            TokenCount: 0,
            CommitCount: 0,
            Files:      make(map[string]int),
            TypesMap:   make(map[string]int),
        }
    }

    // Increment commit count
    contributions[authorKey].CommitCount++

    for _, filePatch := range patch.FilePatches() {
        if filePatch.IsBinary() {
            continue
        }

        // Get file name from filePatch
        from, to := filePatch.Files()
        var fileName string
        if to != nil {
            fileName = to.Path()
        } else if from != nil {
            fileName = from.Path()
        } else {
            fileName = "unknown"
        }

        fileTokens := 0
        for _, chunk := range filePatch.Chunks() {
            if chunk.Type() == diff.Add {
                tokens := tokenize(chunk.Content())
                fileTokens += len(tokens)
                contributions[authorKey].TokenCount += len(tokens)
            }
        }

        // Add file tokens
        if fileTokens > 0 {
            if _, ok := contributions[authorKey].Files[fileName]; !ok {
                contributions[authorKey].Files[fileName] = 0
            }
            contributions[authorKey].Files[fileName] += fileTokens
        }
    }
    return nil
}

// displayCombinedResults formats and prints the combined contribution data.
func displayCombinedResults(result *CommitAnalysisResult, includedNonConventional map[string]bool, nonConventionalInfo []CommitInfo) {
    fmt.Printf("\n--- Git Repository Contribution Analysis ---\n")

    // Overview stats
    duration := result.EndDate.Sub(result.StartDate).Hours() / 24
    overview := []string{
        fmt.Sprintf("Analysis Period: %s to %s (%.0f days)",
            result.StartDate.Format("2006-01-02"),
            result.EndDate.Format("2006-01-02"),
            duration),
        fmt.Sprintf("Total Commits Analyzed: %d",
            result.ConventionalCommits - result.IgnoredTypesTotal + result.SelectedNonConventional),
        fmt.Sprintf("Conventional Commits: %d (ignoring %d)",
            result.ConventionalCommits - result.IgnoredTypesTotal,
            result.IgnoredTypesTotal),
        fmt.Sprintf("Non-Conventional Commits: %d (included %d of %d)",
            result.SelectedNonConventional,
            result.SelectedNonConventional,
            result.NonConventionalCommits),
        fmt.Sprintf("Total Tokens: %d", result.TotalTokens),
    }

    // Add ignored types breakdown if any were ignored
    if result.IgnoredTypesTotal > 0 {
        ignoreDetails := "Ignored Types: "
        for t, count := range result.IgnoredTypesCount {
            ignoreDetails += fmt.Sprintf("%s (%d) ", t, count)
        }
        overview = append(overview, ignoreDetails)
    }

    // Format overview with gum style - FIX: removed border-title flag
    printStyledBox(strings.Join(overview, "\n"), "212")

    // Sort authors for consistent output
    authorEmails := make([]string, 0, len(result.AuthorContributions))
    for email := range result.AuthorContributions {
        authorEmails = append(authorEmails, email)
    }

    // Sort by token count descending, then name ascending
    sort.Slice(authorEmails, func(i, j int) bool {
        email1 := authorEmails[i]
        email2 := authorEmails[j]
        contrib1 := result.AuthorContributions[email1]
        contrib2 := result.AuthorContributions[email2]
        if contrib1.TokenCount != contrib2.TokenCount {
            return contrib1.TokenCount > contrib2.TokenCount
        }
        return contrib1.Name < contrib2.Name
    })

    // Build main contribution table
	outputLines := []string{}

	header := fmt.Sprintf("%-25s %-25s %8s %8s %8s %7s",
		"Author", "Email", "Tokens", "Commits", "Files", "%")
	separator := strings.Repeat("-", 88)
	outputLines = append(outputLines, header, separator)

	totalTokens := result.TotalTokens
	totalCommits := 0
	totalFiles := make(map[string]struct{})

	// Count total commits and unique files
	for _, email := range authorEmails {
		contrib := result.AuthorContributions[email]
		totalCommits += contrib.CommitCount
		for fileName := range contrib.Files {
			totalFiles[fileName] = struct{}{}
		}
	}

	// Build contribution rows
	for i, email := range authorEmails { // Use index 'i' to check if it's the last author
		contrib := result.AuthorContributions[email]
		percentage := 0.0
		if totalTokens > 0 {
			percentage = (float64(contrib.TokenCount) / float64(totalTokens)) * 100
		}

		fileCount := len(contrib.Files)

		line := fmt.Sprintf("%-25s %-25s %8d %8d %8d %6.2f%%",
			truncateString(contrib.Name, 24),
			truncateString(contrib.Email, 24),
			contrib.TokenCount,
			contrib.CommitCount,
			fileCount,
			percentage,
		)
		outputLines = append(outputLines, line)

		// Add commit type breakdown if available
		if len(contrib.TypesMap) > 0 {
			typeBreakdown := "  Types: "
			types := make([]string, 0, len(contrib.TypesMap))
			for t := range contrib.TypesMap {
				types = append(types, t)
			}
			sort.Strings(types)

			for _, t := range types {
				typeBreakdown += fmt.Sprintf("%s (%d) ", t, contrib.TypesMap[t])
			}
			outputLines = append(outputLines, typeBreakdown)
		}

		// Add file breakdown for top 3 files
		if len(contrib.Files) > 0 {
			// Sort files by token count
			files := make([]struct {
				name  string
				count int
			}, 0, len(contrib.Files))

			for name, count := range contrib.Files {
				files = append(files, struct {
					name  string
					count int
				}{name, count})
			}

			sort.Slice(files, func(i, j int) bool {
				return files[i].count > files[j].count
			})

			// Show top 3 files
			maxFiles := 3
			if len(files) < maxFiles {
				maxFiles = len(files)
			}

			fileBreakdown := "  Top files: "
			if contrib.TokenCount > 0 { // Avoid division by zero if token count is somehow 0
				for i := 0; i < maxFiles; i++ {
					f := files[i]
					filePercentage := (float64(f.count) / float64(contrib.TokenCount)) * 100
					fileBreakdown += fmt.Sprintf("%s (%d, %.1f%%) ",
						truncateString(filepath.Base(f.name), 15),
						f.count,
						filePercentage)
				}
			} else if maxFiles > 0 { // Handle case where token count is 0 but files exist
			    for i := 0; i < maxFiles; i++ {
					f := files[i]
					fileBreakdown += fmt.Sprintf("%s (%d) ",
						truncateString(filepath.Base(f.name), 15),
						f.count)
				}
			}

			outputLines = append(outputLines, fileBreakdown)
		}

		// *** ADDED: Add a blank line for separation unless it's the last author ***
		if i < len(authorEmails)-1 {
			outputLines = append(outputLines, "") // Add an empty line
		}
		// *** END OF ADDITION ***
	}

	outputLines = append(outputLines, separator)
	outputLines = append(outputLines, fmt.Sprintf("%-25s %25s %8d %8d %8d %6.2f%%",
		"Total", "", totalTokens, totalCommits, len(totalFiles), 100.0))

	// Try using gum style for the table - FIX: removed border-title flag
	printStyledBox(strings.Join(outputLines, "\n"), "39")
    // Display non-conventional commits that were included
    if len(includedNonConventional) > 0 {
        nonConventionalLines := []string{
            "Included Non-Conventional Commits:",
            separator,
        }

        for hash, included := range includedNonConventional {
            if included {
                var commitInfo CommitInfo
                for _, info := range nonConventionalInfo {
                    if info.Hash == hash {
                        commitInfo = info
                        break
                    }
                }
                nonConventionalLines = append(nonConventionalLines,
                    fmt.Sprintf("[%s] %s (%s <%s>)",
                    hash,
                    commitInfo.Message,
                    commitInfo.Author,
                    commitInfo.Email))
            }
        }

        if len(nonConventionalLines) > 2 {
            // FIX: removed border-title flag
            printStyledBox(strings.Join(nonConventionalLines, "\n"), "208")
        }
    }
}

// printStyledBox is a replacement for runGumStyle that doesn't use the incompatible border-title flag
func printStyledBox(text, foreground string) {
    styleArgs := []string{
        "style",
        "--border", "rounded",
        "--padding", "1 2",
        "--border-foreground", foreground,
    }

    styleArgs = append(styleArgs, text)

    cmd := exec.Command("gum", styleArgs...)
    cmd.Stdout = os.Stdout
    cmd.Stderr = os.Stderr
    err := cmd.Run()
    if err != nil {
        fmt.Println("\n(gum style command failed, printing plain text)")
        fmt.Println(text)
    }
}

// runGumChoose executes 'gum choose' with enhanced options
// preSelect allows specifying items to be pre-selected (if any)
// allowEmpty determines whether the user can select 0 items
func runGumChoose(options []string, prompt string, allowPreSelect bool, preSelect []string) ([]string, error) {
    if len(options) == 0 {
        return []string{}, nil
    }

    args := []string{
        "choose",
        "--no-limit",
        "--header", prompt,
        "--cursor", "> ",
        "--selected-prefix", "[✓] ",
        "--unselected-prefix", "[ ] ",
    }

    // Add pre-selection if requested and items are provided
    if allowPreSelect && preSelect != nil && len(preSelect) > 0 {
        for _, item := range preSelect {
            if contains(options, item) {
                args = append(args, "--selected", item)
            }
        }
    }

    args = append(args, options...)

    cmd := exec.Command("gum", args...)
    var out strings.Builder
    cmd.Stdout = &out
    cmd.Stderr = os.Stderr
    cmd.Stdin = os.Stdin

    err := cmd.Run()
    if err != nil {
        // Check for cancellation
        if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 130 {
            log.Println("gum choose cancelled by user.")
            return nil, fmt.Errorf("gum choose command cancelled")
        } else if ok {
            log.Printf("gum choose exited with error code %d", exitErr.ExitCode())
        }
        return nil, fmt.Errorf("gum choose command failed: %w", err)
    }

    selected := strings.TrimSpace(out.String())
    if selected == "" {
        return []string{}, nil // Nothing selected, return empty slice
    }

    return strings.Split(selected, "\n"), nil
}

// runGumChooseWithIndices executes 'gum choose' and returns the indices of selected items
// This is helpful when we need to know which items were selected by position
func runGumChooseWithIndices(options []string, prompt string, allowPreSelect bool, preSelect []int) ([]int, error) {
    // Allow returning an empty selection
    selected, err := runGumChoose(options, prompt, true, nil) // We handle preselect differently
    if err != nil {
        return nil, err
    }

    // Convert selected items back to indices
    selectedIndices := []int{}
    for _, item := range selected {
        for i, option := range options {
            if item == option {
                selectedIndices = append(selectedIndices, i)
                break
            }
        }
    }

    return selectedIndices, nil
}

// helper function: check if a slice contains a string
func contains(slice []string, item string) bool {
    for _, s := range slice {
        if s == item {
            return true
        }
    }
    return false
}

// tokenize splits a line into simplified tokens.
func tokenize(line string) []string {
    processedLine := tokenRegex.ReplaceAllString(line, " ")
    processedLine = strings.TrimSpace(processedLine)
    if processedLine == "" {
        return []string{}
    }
    return strings.Fields(processedLine)
}

// Helper function (unused but potentially useful)
func isPunctOrSymbol(r rune) bool {
    return unicode.IsPunct(r) || unicode.IsSymbol(r)
}

// truncateString truncates a string if it's longer than the specified length
func truncateString(s string, maxLen int) string {
    if len(s) <= maxLen {
        return s
    }
    return s[:maxLen-3] + "..."
}
