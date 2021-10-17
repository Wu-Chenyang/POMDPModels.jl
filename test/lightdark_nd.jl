using Test
using POMDPModels

let
    rng = MersenneTwister(41)
    N = 5
    p = LightDarkND{N, typeof(default_sigma)}()
    @test discount(p) == 0.95
    s0 = LightDarkNDState{N}(0,fill(0.0, N))
    s0, _, r = @gen(:sp, :o, :r)(p, s0, +1, rng)
    @test s0.loc[1] == 1.0
    @test r == 0
    s1, _, r = @gen(:sp, :o, :r)(p, s0, 0, rng)
    @test s1.status < 0
    @test r == -10.0
    s2 = LightDarkNDState{N}(0, fill(5.0, N))
    obs = rand(rng, observation(p, nothing, nothing, s2))
    @test sum(abs.(obs.-5.0)) <= 0.1

    sv = convert_s(Array{Float64}, s2, p)
    @time for i = 1:1000000 convert_s(Array{Float64}, s2, p) end
    @test sv == [0.0, fill(5.0, N)...]
    s = convert_s(LightDarkNDState{N}, sv, p)
    @time for i = 1:1000000 convert_s(LightDarkNDState{N}, sv, p) end
    @test s == s2
end
