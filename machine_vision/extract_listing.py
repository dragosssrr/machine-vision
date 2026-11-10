"""extract_listing.py -- copy the code between '# [slide:begin]' and '# [slide:end]' of example_pipeline.py
to figures/generated/listing_track_a.py and listing_track_b.py (split at '# [slide:split]'), so the slides
always show the tested code.  Run: python3 extract_listing.py"""
import os
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, "example_pipeline.py")).read().splitlines()
a = next(i for i, l in enumerate(src) if "[slide:begin]" in l) + 1
m = next(i for i, l in enumerate(src) if "[slide:split]" in l)
b = next(i for i, l in enumerate(src) if "[slide:end]" in l)
out = os.path.join(here, "..", "figures", "generated")
for name, lines in (("a", src[a:m]), ("b", src[m + 1:b])):
    open(os.path.join(out, f"listing_track_{name}.py"), "w").write("\n".join(l.rstrip() for l in lines) + "\n")
    print(f"listing_track_{name}.py: {len(lines)} lines, longest {max(len(l.rstrip()) for l in lines)} characters")

# the T4 animation listing (animate_relative_pose.py)
src = open(os.path.join(here, "animate_relative_pose.py")).read().splitlines()
a = next(i for i, l in enumerate(src) if "[slide:begin]" in l) + 1
b = next(i for i, l in enumerate(src) if "[slide:end]" in l)
open(os.path.join(out, "listing_animate.py"), "w").write("\n".join(l.rstrip() for l in src[a:b]) + "\n")
print(f"listing_animate.py: {b - a} lines, longest {max(len(l.rstrip()) for l in src[a:b])} characters")
