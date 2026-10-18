# Verify the imported Markdown tables against the rendered site, including tables
# inside collapsed details. Run after `bundle exec jekyll build`.
require 'json'
require 'nokogiri'

repo = File.expand_path('..', __dir__)
site = File.expand_path(ARGV.fetch(0, '_site'), repo)
posts = Dir[File.join(__dir__, 'imports', '*.json')].flat_map do |manifest|
  JSON.parse(File.read(manifest)).fetch('posts')
end.uniq { |post| post.fetch('post') }

def table_cells(line)
  line.strip.sub(/^\|/, '').sub(/\|$/, '').split(/(?<!\\)\|/, -1)
end

def markdown_tables(source)
  lines = source.lines
  tables = []
  fence = nil
  index = 0
  while index < lines.length
    line = lines[index]
    if (marker = line.match(/^\s*(`{3,}|~{3,})/))
      fence = fence ? nil : marker[1][0]
      index += 1
      next
    end
    if !fence && line.lstrip.start_with?('|') &&
       lines[index + 1]&.match?(/^\s*\|[\s:|\-]+\|\s*$/)
      start = index
      rows = []
      while index < lines.length && lines[index].lstrip.start_with?('|')
        rows << lines[index]
        index += 1
      end
      tables << { line: start + 1, rows: rows.length - 2,
                  columns: table_cells(rows.first).length }
    else
      index += 1
    end
  end
  tables
end

errors = []
total = 0
posts.each do |post|
  source_path = File.join(repo, post.fetch('post'))
  output_path = File.join(site, post.fetch('url'), 'index.html')
  unless File.file?(output_path)
    errors << "#{post['post']}: missing built page #{output_path}"
    next
  end
  expected = markdown_tables(File.read(source_path))
  body = Nokogiri::HTML(File.read(output_path)).at_css('.writing-body')
  actual = body ? body.css('table') : []
  total += expected.length
  if actual.length != expected.length
    errors << "#{post['post']}: #{expected.length} source tables, #{actual.length} rendered tables"
    next
  end
  expected.zip(actual).each do |table, rendered|
    location = "#{post['post']}:#{table[:line]}"
    rows = rendered.css('tbody > tr')
    errors << "#{location}: missing or extra data rows" unless rows.length == table[:rows]
    cells = rendered.css('thead > tr > th').length
    errors << "#{location}: incorrect header columns" unless cells == table[:columns]
    rows.each_with_index do |row, row_index|
      unless row.css('td').length == table[:columns]
        errors << "#{location}: incorrect columns in row #{row_index + 1}"
      end
    end
  end
end

abort errors.join("\n") unless errors.empty?
puts "Verified #{total} tables in #{posts.length} posts, including collapsed sections."
