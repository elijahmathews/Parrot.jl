#
# Parrot.jl/src/prospect.jl
#
# By Elijah Mathews (me@elijahmathews.com)
# openpgp4fpr:ac1d3fb1e8a5eb7d14bd587b2932c725055a90d8
#

using PyCall

"""
    buildobs(...)

Generate a observation dictionary for the purposes of generating
training and validation data from [Prospector](https://github.com/bd-j/prospector).
"""
function buildobs(; snr=10, ldist=10, extras...)

    # The obs dictionary, empty for now.
    obs = Dict()

    # We will not use any obs filters for now.
    obs["filters"] = nothing

    # Since no filters, set "phot_wave" to nothing too.
    obs["phot_wave"] = nothing

    # No observations here, so set these to nothing.
    obs["wavelength"] = nothing
    obs["spectrum"] = nothing
    obs["unc"] = nothing
    obs["mask"] = nothing

    return obs

end

"""
    buildmodel(...)

Generate an SedModel object from [Prospector](https://github.com/bd-j/prospector)
for the purposes of generating training and validation data.
"""
function buildmodel(; object_redshift = nothing, ldist=10, fixed_metallicity = nothing, add_duste = false, extras...)

    # Required Python packages.
    sedmodel = pyimport("prospect.models.sedmodel")

    # Set template to parametric star formation history.
    py"""

    from prospect.models.templates import TemplateLibrary

    def get_template_library(key):

        return TemplateLibrary[key]

    """
    modelparams = py"get_template_library"("parametric_sfh")

    # Change to fixed stellar metallicity.
    modelparams["logzsol"]["isfree"] = false
    modelparams["logzsol"]["init"] = 0

    # Change to fixed redshift.
    modelparams["zred"]["isfree"] = false
    modelparams["zred"]["init"] = 0

    # Change to fixed stellar mass formed.
    modelparams["mass"]["isfree"] = false
    modelparams["mass"]["init"] = 1

    # Instantiate model object using modelparams.
    model = sedmodel.SedModel(modelparams)

    return model, modelparams

end

"""
    buildsps(...)

Generate a CSPSpecBasis object for the purposes of generating training
and validation data from [Prospector](https://github.com/bd-j/prospector).
"""
function buildsps(; zcontinuous = 1, extras...)

    sources = pyimport("prospect.sources")

    sps = sources.CSPSpecBasis(; zcontinuous=zcontinuous, extras...)

    return sps

end

"""
    generatedata(amount::Integer)

Generate training and validation data from [Prospector](https://github.com/bd-j/prospector)
(using [FSPS](https://github.com/cconroy20/fsps)). Returns a tuple:

    (λarray, params, Iarray)

where `λarray` is an `amount × 1` array consisting of the wavelengths in the spectra
(where `M` is the number of wavelengths), `params` is an `amount × 3` array consisting
of stellar population parameters (first column is the diffuse dust coefficient
`dust2`, second column is the universe age at galaxy's lookback time `tage`, and
the third column is the star formation timescale `tau` for an exponentially
declining star formation history), and `Iarray` is an `amount × M` array consisting
of the `log10` spectra in units of maggies.
"""
function generatedata(amount::Integer)

    # Generate requisite model objects.
    obs = buildobs()
    model, modelparams = buildmodel()
    sps = buildsps()

    # Grab wavelengths.
    λarray = sps.wavelengths

    # Initialize output arrays.
    farray = zeros(amount, length(λarray))
    params = zeros(amount, 3)

    # Loop to generate data.
    for i in 1:amount

        # Sample the stellar population parameters and save them.
        params[i,1] = modelparams["dust2"]["prior"].sample()[1]
        params[i,2] = modelparams["tage"]["prior"].sample()[1]
        params[i,3] = modelparams["tau"]["prior"].sample()[1]

        # Construct the array θ of stellar population parameters.
        θ = params[i,:]

        # Generate the spectrum and other stuff.
        spec, _, _ = model.sed(θ; obs = obs, sps = sps)

        # Take log10(...) of the spetrum.
        farray[i,:] = log10.(spec)

    end

    return λarray, params, farray

end
