/**
 * Windows PowerShell writes to a pipe in the console's OEM code page (cp866 on a Russian Windows, cp437 on an English one),
 * not in UTF-8, while Node reads the pipe as UTF-8 - so a Cyrillic folder name came back as mojibake or as question marks (#3202).
 * Every script whose output can hold a folder, file or font name starts with this line, before anything is written.
 */
export const POWERSHELL_UTF8 = '[Console]::OutputEncoding = New-Object System.Text.UTF8Encoding $false; $OutputEncoding = [Console]::OutputEncoding; ';

/** The script with the UTF-8 line in front of it. */
export const utf8Script = script => POWERSHELL_UTF8 + script;
